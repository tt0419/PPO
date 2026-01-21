# 状態空間の分析と改善提案

**日付**: 2024年12月
**ステータス**: 研究室の仲間からの指摘に基づく分析

---

## 第1章: 現在の状態空間の問題点

### 1.1 状態空間の内訳

| カテゴリ | 次元数 | 割合 | 内容 |
|----------|--------|------|------|
| 救急車特徴量 | **960** | **96.1%** | 192台 × 5特徴量 |
| 事案特徴量 | 10 | 1.0% | 位置、傷病度、待機時間など |
| 時間特徴量 | 8 | 0.8% | エピソード進行度、時刻など |
| 空間特徴量 | 21 | 2.1% | 移動時間統計、カバレッジ |
| **合計** | **999** | 100% | |

### 1.2 救急車特徴量の詳細（問題の中心）

各救急車について5次元:
1. 位置x（緯度）← **冗長**: 移動時間があれば不要
2. 位置y（経度）← **冗長**: 移動時間があれば不要
3. 状態（利用可能/出動中）← 必要
4. 出動回数 ← **不要**: 配車判断に関係なし
5. 事案までの移動時間 ← **必要**: 最重要

### 1.3 問題点のまとめ

```
問題1: 救急車特徴量が支配的（96%）
  → 本当に必要な情報（傷病度、移動時間統計）が埋もれる

問題2: 冗長な情報
  - 位置情報: 384次元（移動時間があれば不要）
  - 出動回数: 192次元（配車判断に無関係）
  - 出動中救急車: 選択不可なのに情報を持つ

問題3: 次元の呪い
  - 999次元 × 8000エピソード = 学習困難
  - CartPole: 4次元 × 数百エピソード = 学習容易
  - 現在の設定は約25倍の複雑さ

問題4: 状態と行動の非対応
  - 状態: 救急車の情報が「位置」でエンコード
  - 行動: 救急車の「番号」で選択
  - PPOは位置と番号の対応を学習する必要 → 非効率
```

---

## 第2章: 他の強化学習タスクとの比較

| 環境 | 状態次元 | 行動次元 | 学習量 | 結果 |
|------|----------|----------|--------|------|
| CartPole | 4 | 2 | 500-1000 ep | 成功 |
| LunarLander | 8 | 4 | 1000-3000 ep | 成功 |
| Pendulum | 3 | 1 | 数千 ep | 成功 |
| ロボットアーム | 20-50 | 4-7 | 数万 ep | 成功 |
| Atari (DQN) | 28,224 | 4-18 | 数百万 step | 成功（CNN使用） |
| **現在のEMS** | **999** | **192** | 8000 ep | **困難** |

**重要な観察**:
- Atariは高次元だが、CNNで特徴抽出している
- 現在のEMSは生の特徴量を直接使用 → 学習困難

---

## 第3章: 改善提案

### 3.1 推奨案: Top-K設計（状態37次元、行動10次元）

**コンセプト**: 
- 192台全てではなく、移動時間が短いTop-K台のみを考慮
- 傷病度考慮運用も時間制限で候補を絞っている → 同等のアプローチ

**状態空間（37次元）**:

```
[0-1] 傷病度（2次元）
  - is_severe: 重症系フラグ（重症/重篤/死亡 = 1）
  - is_mild: 軽症系フラグ（軽症/中等症 = 1）

[2-31] Top-10救急車の特徴量（30次元）
  各救急車について3次元:
  - travel_time: 移動時間（正規化、0-30分を0-1に）
  - coverage_loss: この救急車を出した場合のカバレッジ損失（0-1）
  - is_returning: 帰署中フラグ（オプション、または署からの距離）

[32-36] グローバル統計（5次元）
  - available_count: 利用可能救急車数（正規化）
  - coverage_rate: 現在のカバレッジ率
  - time_of_day: 時刻（0-1）
  - within_6min_count: 6分以内到達可能台数（正規化）
  - avg_travel_time: Top-10の平均移動時間（正規化）
```

**行動空間（10次元）**:

```
action = 0: Top-1（最短移動時間）を選択 = 直近隊
action = 1: Top-2を選択
action = 2: Top-3を選択
...
action = 9: Top-10を選択
```

### 3.2 PPOの学習目標

```
重症系（is_severe = 1）:
  目標: action = 0（最短移動時間）を選ぶ
  報酬: -応答時間
  
軽症系（is_mild = 1）:
  目標: カバレッジを考慮して選ぶ
  報酬: -応答時間 + カバレッジボーナス
  
  PPOが学習すべきこと:
  「Top-1は移動時間が最短だが、カバレッジ損失が大きい場合、
   Top-2やTop-3を選んだ方が全体として良い」
```

### 3.3 比較表

| 項目 | 現在 | 提案 | 改善 |
|------|------|------|------|
| 状態次元 | 999 | 37 | **96%削減** |
| 行動次元 | 192 | 10 | **95%削減** |
| 学習複雑さ | 192クラス分類 | 10クラス分類 | **19倍簡略化** |
| 必要エピソード数 | 50,000+（推定） | 5,000（推定） | **10分の1** |
| 8000 epでの学習 | △ 困難 | ◎ 十分 | **大幅改善** |

---

## 第4章: 実装の変更点

### 4.1 state_encoder.py の変更

```python
class CompactStateEncoder:
    """コンパクトな状態エンコーダー（37次元）"""
    
    def __init__(self, config: Dict, top_k: int = 10):
        self.top_k = top_k
        self.state_dim = 2 + (top_k * 3) + 5  # 37次元
        
    def encode_state(self, state_dict: Dict) -> np.ndarray:
        features = np.zeros(self.state_dim, dtype=np.float32)
        
        incident = state_dict.get('pending_call')
        ambulances = state_dict['ambulances']
        
        # 1. 傷病度（2次元）
        if incident:
            severity = incident.get('severity', '')
            features[0] = 1.0 if is_severe_condition(severity) else 0.0
            features[1] = 1.0 if is_mild_condition(severity) else 0.0
        
        # 2. Top-K救急車の情報（30次元）
        top_k_info = self._get_top_k_ambulances(ambulances, incident)
        for i, amb_info in enumerate(top_k_info[:self.top_k]):
            base_idx = 2 + i * 3
            features[base_idx] = amb_info['travel_time'] / 30.0  # 30分で正規化
            features[base_idx + 1] = amb_info['coverage_loss']
            features[base_idx + 2] = amb_info['station_distance'] / 10.0  # 10kmで正規化
        
        # 3. グローバル統計（5次元）
        global_idx = 2 + self.top_k * 3
        available_count = sum(1 for a in ambulances.values() if a['status'] == 'available')
        features[global_idx] = available_count / 192.0
        features[global_idx + 1] = self._calculate_coverage_rate(ambulances)
        features[global_idx + 2] = state_dict.get('time_of_day', 12) / 24.0
        features[global_idx + 3] = self._count_within_6min(top_k_info) / self.top_k
        features[global_idx + 4] = np.mean([a['travel_time'] for a in top_k_info]) / 30.0
        
        return features
    
    def _get_top_k_ambulances(self, ambulances, incident):
        """移動時間順にTop-K救急車を取得"""
        if incident is None:
            return [{'travel_time': 30, 'coverage_loss': 0, 'station_distance': 0}] * self.top_k
        
        incident_h3 = incident.get('h3_index')
        candidates = []
        
        for amb_id, amb_state in ambulances.items():
            if amb_state['status'] != 'available':
                continue
            
            travel_time = self._calculate_travel_time(amb_state['current_h3'], incident_h3)
            coverage_loss = self._calculate_coverage_loss(amb_id, ambulances)
            station_distance = self._calculate_station_distance(amb_state)
            
            candidates.append({
                'amb_id': amb_id,
                'travel_time': travel_time / 60.0,  # 秒→分
                'coverage_loss': coverage_loss,
                'station_distance': station_distance
            })
        
        # 移動時間順にソート
        candidates.sort(key=lambda x: x['travel_time'])
        
        # Top-Kに足りない場合はダミーで埋める
        while len(candidates) < self.top_k:
            candidates.append({'travel_time': 30, 'coverage_loss': 0, 'station_distance': 0})
        
        return candidates[:self.top_k]
```

### 4.2 ems_environment.py の変更

```python
class EMSEnvironment:
    def __init__(self, config_path: str):
        # ...
        self.top_k = config.get('top_k_ambulances', 10)
        self.action_dim = self.top_k  # 192 → 10に変更
        
        # コンパクトエンコーダーを使用
        self.state_encoder = CompactStateEncoder(config, self.top_k)
        self.state_dim = self.state_encoder.state_dim  # 37
        
        # Top-K救急車のIDを保持（actionを実際の救急車IDに変換するため）
        self.current_top_k_ids = []
    
    def _get_observation(self) -> np.ndarray:
        """状態を取得（Top-K情報も更新）"""
        state_dict = self._build_state_dict()
        
        # Top-KのIDを更新
        self.current_top_k_ids = self._get_top_k_ambulance_ids()
        
        return self.state_encoder.encode_state(state_dict)
    
    def _get_top_k_ambulance_ids(self) -> List[int]:
        """移動時間順にTop-K救急車のIDを取得"""
        if self.pending_call is None:
            return list(range(self.top_k))
        
        candidates = []
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_state['status'] != 'available':
                continue
            
            travel_time = self._calculate_travel_time(
                amb_state['current_h3'],
                self.pending_call['h3_index']
            )
            candidates.append((amb_id, travel_time))
        
        # 移動時間順にソート
        candidates.sort(key=lambda x: x[1])
        
        # Top-KのIDを返す
        return [c[0] for c in candidates[:self.top_k]]
    
    def step(self, action: int) -> StepResult:
        """行動を実行（actionはTop-K内のインデックス）"""
        
        # actionをTop-K内のインデックスとして解釈
        if action < len(self.current_top_k_ids):
            actual_ambulance_id = self.current_top_k_ids[action]
        else:
            # フォールバック: Top-1を選択
            actual_ambulance_id = self.current_top_k_ids[0] if self.current_top_k_ids else 0
        
        # 実際の配車処理
        dispatch_result = self._dispatch_ambulance(actual_ambulance_id, ...)
        
        # ...
    
    def get_action_mask(self) -> np.ndarray:
        """行動マスク（Top-Kは基本的に全て利用可能）"""
        mask = np.ones(self.action_dim, dtype=bool)
        
        # Top-Kに満たない場合は無効化
        valid_count = len(self.current_top_k_ids)
        if valid_count < self.action_dim:
            mask[valid_count:] = False
        
        return mask
```

### 4.3 config.yaml の変更

```yaml
# 状態空間設定
state_encoding:
  mode: "compact"  # "full" (999次元) or "compact" (37次元)
  top_k_ambulances: 10
  
  # Top-K救急車の特徴量
  ambulance_features:
    - travel_time      # 移動時間
    - coverage_loss    # カバレッジ損失
    - station_distance # 署からの距離

# 行動空間設定（Top-Kモードでは自動設定）
# action_dim は top_k_ambulances と同じ値になる

# PPO設定（調整推奨）
ppo:
  n_episodes: 8000
  learning_rate:
    actor: 0.0003
    critic: 0.001
  # 行動空間が小さいので、探索を増やす
  entropy_coef: 0.02  # 0.01 → 0.02
```

---

## 第5章: 期待される効果

### 5.1 学習効率

```
現在: 999次元 × 192行動 × 8000エピソード → 学習困難
提案: 37次元 × 10行動 × 8000エピソード → 学習可能

改善率:
  - 状態空間: 27分の1
  - 行動空間: 19分の1
  - 総複雑さ: 約500分の1
```

### 5.2 傷病度考慮運用との比較

```
直近隊運用:
  → action = 0（Top-1）を常に選択するのと同等

傷病度考慮運用:
  重症系: action = 0
  軽症系: カバレッジ損失が低いTop-Kから選択
  
  → PPOがこのロジックを学習できる可能性が高い
```

### 5.3 解釈可能性

```
PPOの出力を分析:
  action = 0 が多い → 直近隊運用に近い
  action > 0 が多い → カバレッジ考慮を学習した
  
傷病度別の分析:
  重症系でaction = 0 → 正しい（最短優先）
  軽症系でaction > 0 → カバレッジ考慮
```

---

## 第6章: 実験計画

### 6.1 検証実験

```
実験1: 学習曲線の比較
  - 現在のモデル（999次元）vs 提案モデル（37次元）
  - 同じ学習期間、同じハイパーパラメータ
  - 期待: 提案モデルの学習が圧倒的に速い

実験2: 傷病度考慮運用との比較
  - 重症率の高い期間でテスト
  - 期待: 傷病度考慮運用に近い性能

実験3: 行動パターンの分析
  - action = 0 の選択率を傷病度別に分析
  - 期待: 重症系でaction=0が多い、軽症系でaction>0も選択
```

### 6.2 ハイパーパラメータ調整

```
調整項目:
  - top_k: 5, 10, 15, 20（推奨: 10）
  - entropy_coef: 0.01, 0.02, 0.05（探索促進）
  - learning_rate: 現行維持（0.0003/0.001）
  - n_episodes: 4000, 8000, 12000（学習量）
```

---

## 結論

### 問題の根本原因

```
999次元の状態空間は、PPOが8000エピソードで学習するには大きすぎる。
特に、192台の救急車それぞれに5次元（=960次元）を割り当てているのが問題。
```

### 推奨する解決策

```
Top-K設計:
  - 状態: 37次元（96%削減）
  - 行動: 10次元（95%削減）
  - 学習複雑さ: 約500分の1

これにより:
  - 8000エピソードで十分な学習が可能
  - 傷病度考慮運用のロジックを学習できる可能性
  - 学習結果の解釈が容易
```

### 実装優先度

```
1. state_encoder.py の CompactStateEncoder 実装
2. ems_environment.py の action_dim 変更と step() 修正
3. config.yaml の設定追加
4. 検証実験の実施
```

---

*作成日: 2024年12月*
*分析: 状態空間999次元の問題点と改善提案*
