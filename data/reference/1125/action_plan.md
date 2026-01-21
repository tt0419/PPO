# ===============================================
# 改善実装のアクションプラン
# ===============================================

## 📊 現状の確認

### 学習曲線の評価 ✅
- actor_loss: 安定（0.05-0.1付近）
- critic_loss: 収束
- entropy: 緩やかに減少
→ **ep3000で十分収束している**

### 性能評価
```
ep3000-重症重視×アクティブ:
  重症系平均RT: 10.98分
  重症系6分達成率: 19.5%

直近隊運用:
  重症系平均RT: 10.87分（-0.11分）
  重症系6分達成率: 20.8%（+1.3%）

傷病度考慮運用:
  重症系平均RT: 10.58分（-0.40分）← 目標
  重症系6分達成率: 24.1%（+4.6%）← 目標
```

### 問題点
**PPOが直近隊に勝てない根本原因**:
- 報酬設計が「応答時間のみ」に偏っている
- カバレッジの概念がない
- 傷病度考慮運用のような「バランス」がない

---

## 🎯 改善の核心アイデア

**「傷病度考慮運用のロジックをPPOの報酬設計に組み込む」**

```python
# 傷病度考慮運用の成功要因
重症系: 応答時間のみ（直近隊と同じ）
軽症系: score = time × 0.6 + coverage_loss × 0.4

# PPO（現在）
reward = -time × weight + bonus
# ↓ カバレッジが考慮されていない

# PPO（改良版）
重症系: reward = -time × 3.0 + bonus
軽症系: reward = -time × 0.6 - coverage_loss × 0.4 + bonus
# ↑ 傷病度考慮運用と同じロジック
```

---

## 🚀 実装の優先順位

### Phase 1: 最小限の実装（推奨度：★★★★★）

**目的**: カバレッジ損失を報酬に組み込む効果を確認

**所要時間**: 1-2日

**実装内容**:

1. **カバレッジ損失の計算関数を追加**
   ```python
   # reinforcement_learning/environment/ems_environment.py
   
   def calculate_coverage_loss(self, selected_ambulance_id, 
                               available_ambulances, request_h3):
       """カバレッジ損失を計算（0-1の範囲）"""
       # 実装は implementation_guide.md を参照
       pass
   ```

2. **報酬計算ロジックの修正**
   ```python
   # reinforcement_learning/environment/ems_environment.py
   # step()メソッド内
   
   if severity in ['重症', '重篤', '死亡']:
       # 重症系: 応答時間のみ
       reward = -rt_minutes * 3.0 + bonus
   else:
       # 軽症系: 応答時間 + カバレッジ
       time_component = -rt_minutes * weight * 0.6
       coverage_loss = self.calculate_coverage_loss(...)
       coverage_component = -coverage_loss * 100.0 * 0.4
       reward = time_component + coverage_component + bonus
   ```

3. **簡単なテスト実行**
   ```bash
   # ep1000で効果を確認
   python train_ppo.py --config config_coverage_aware_v1.yaml \
       --episodes 1000
   ```

**期待される効果**:
- カバレッジを考慮した配車が学習される
- 軽症系の配車パターンが改善
- 重症系の性能が維持される

**判断基準**:
- ✅ 学習が安定している
- ✅ 軽症系でカバレッジ損失が減少傾向
- ✅ 重症系の性能が悪化していない
→ Phase 2へ

---

### Phase 2: 完全版の実装（推奨度：★★★★☆）

**目的**: 傷病度考慮運用レベルの性能を達成

**所要時間**: 2-3日

**実装内容**:

1. **アクションマスクの強化**
   ```python
   # 軽症系の場合、13分以内 かつ カバレッジ損失が小さい候補のみ
   def _get_action_mask_with_coverage(self, request, available_ambulances):
       if severity in ['軽症', '中等症']:
           for amb_id in available_ambulances:
               response_time = self._calculate_response_time(amb_id, request)
               coverage_loss = self.calculate_coverage_loss(amb_id, ...)
               
               if response_time <= 780 and coverage_loss < 0.8:
                   mask[amb_id] = True
   ```

2. **カリキュラム学習の導入**
   ```yaml
   # config_coverage_aware_v1.yaml に既に記載済み
   curriculum:
     enabled: true
     stages:
       - name: "time_only"  # ep0-1000
         mild_params: {time_weight: 1.0, coverage_weight: 0.0}
       - name: "introduce_coverage"  # ep1000-3000
         mild_params: {time_weight: 0.8, coverage_weight: 0.2}
       - name: "final_balance"  # ep3000-5000
         mild_params: {time_weight: 0.6, coverage_weight: 0.4}
   ```

3. **状態表現の拡張（オプション）**
   ```python
   # カバレッジスコアを状態に追加
   coverage_6min_rate = self._calculate_coverage_rate(360)
   coverage_13min_rate = self._calculate_coverage_rate(780)
   state = np.concatenate([state, [coverage_6min_rate, coverage_13min_rate]])
   ```

4. **本格的な学習**
   ```bash
   python train_ppo.py --config config_coverage_aware_v1.yaml \
       --episodes 5000
   ```

**期待される効果**:
- 傷病度考慮運用に近い性能（重症系10.58分、6分達成率24.1%）
- 直近隊運用を上回る性能
- 学習の安定性向上

**判断基準**:
- ✅ 重症系平均RT < 10.87分（直近隊）
- ✅ 重症系6分達成率 > 20.8%（直近隊）
- ✅ 全体のバランスが保たれている
→ Phase 3へ（または成功）

---

### Phase 3: 最適化（推奨度：★★★☆☆）

**目的**: さらなる性能向上と効率化

**所要時間**: 1-2日

**実装内容**:

1. **カバレッジ損失計算の高速化**
   ```python
   # 事前計算・キャッシュ
   self.coverage_cache = {}
   
   def calculate_coverage_loss_cached(self, amb_id, ...):
       cache_key = (amb_id, timestamp)
       if cache_key in self.coverage_cache:
           return self.coverage_cache[cache_key]
       # 計算してキャッシュ
   ```

2. **ハイパーパラメータチューニング**
   ```yaml
   # 学習率、バッチサイズ、エントロピー係数などを調整
   ppo:
     learning_rate:
       actor: [0.00005, 0.0001, 0.0002]  # グリッドサーチ
     batch_size: [1024, 2048, 4096]
     entropy_coef: [0.01, 0.02, 0.03]
   ```

3. **ネットワークアーキテクチャの調整**
   ```yaml
   network:
     actor:
       hidden_layers: [512, 256, 128]  # より大きく
       attention:
         enabled: true  # Attention機構の導入
   ```

**期待される効果**:
- 傷病度考慮運用を上回る性能
- 学習時間の短縮
- より安定した学習

---

## 📋 実装チェックリスト

### Phase 1 の必須タスク

- [ ] `calculate_coverage_loss()`関数の実装
  - [ ] `_get_coverage_sample_points()`
  - [ ] `_get_min_response_time()`
  - [ ] `_simple_coverage_loss()`
  
- [ ] 報酬計算ロジックの修正
  - [ ] 重症系と軽症系で分岐
  - [ ] カバレッジコンポーネントの追加
  
- [ ] テスト実行（ep1000）
  - [ ] 学習曲線の確認
  - [ ] カバレッジ損失のログ確認
  - [ ] 重症系性能の維持確認

### Phase 2 の推奨タスク

- [ ] アクションマスクの強化
  - [ ] 時間制約の追加
  - [ ] カバレッジ損失閾値の追加
  
- [ ] カリキュラム学習の実装
  - [ ] 設定ファイルの読み込み
  - [ ] エピソード数に応じた重み変更
  
- [ ] 本格的な学習（ep5000）
  - [ ] wandbでモニタリング
  - [ ] 各ステージでの性能確認

### Phase 3 の最適化タスク

- [ ] 高速化
  - [ ] カバレッジ損失のキャッシュ
  - [ ] 並列計算の導入
  
- [ ] チューニング
  - [ ] ハイパーパラメータの調整
  - [ ] ネットワークアーキテクチャの改良

---

## 🎯 期待される最終結果

### 定量的な目標

```
目標（改良版PPO）:
  重症系平均RT: 10.5分以下
  重症系6分達成率: 22%以上
  
  → 直近隊運用（10.87分、20.8%）を上回る

理想（傷病度考慮運用レベル）:
  重症系平均RT: 10.58分
  重症系6分達成率: 24.1%
  
  → 傷病度考慮運用に匹敵
```

### 質的な目標

1. **重症系の優先**
   - 重症系は常に最寄りを選択（直近隊運用と同じ）
   - 応答時間を最小化

2. **軽症系のバランス**
   - 応答時間とカバレッジのバランス（60:40）
   - カバレッジ損失が大きい配車を避ける

3. **学習の安定性**
   - カリキュラム学習により段階的に学習
   - 収束が早く、安定した性能

---

## 💡 重要なポイント

### 成功の鍵

1. **傷病度考慮運用の成功パターンを忠実に模倣**
   - 重症系: 応答時間のみ
   - 軽症系: 応答時間 + カバレッジ

2. **カバレッジ損失の正確な計算**
   - 6分カバレッジと13分カバレッジの変化
   - 重み: 50% + 50%

3. **段階的な導入（カリキュラム学習）**
   - まずは応答時間のみ学習
   - 徐々にカバレッジを導入
   - 最終的なバランスに到達

### リスクと対策

**リスク1: カバレッジ損失の計算コストが高い**
- 対策: サンプルサイズを調整（20ポイント）
- 対策: 必要に応じてキャッシュ化

**リスク2: 学習が不安定になる**
- 対策: カリキュラム学習で段階的に導入
- 対策: 学習率を小さくする

**リスク3: カバレッジを重視しすぎて応答時間が悪化**
- 対策: 重みのバランスを調整（60:40から始める）
- 対策: 時間制約を設ける（13分以内の候補のみ）

---

## 📞 次のステップ

### 今すぐ実行すべきこと

1. **Phase 1 の実装を開始**
   ```bash
   # ステップ1: ブランチを作成
   git checkout -b feature/coverage-aware-reward
   
   # ステップ2: カバレッジ損失関数を実装
   # reinforcement_learning/environment/ems_environment.py を編集
   
   # ステップ3: 報酬計算ロジックを修正
   # 同じファイル内のstep()メソッドを編集
   
   # ステップ4: テスト実行
   python train_ppo.py --config config_coverage_aware_v1.yaml \
       --episodes 1000
   ```

2. **実装ガイドを参照**
   - `/mnt/user-data/outputs/implementation_guide.md`
   - 具体的なコード例が記載されています

3. **設定ファイルを使用**
   - `/mnt/user-data/outputs/config_coverage_aware_v1.yaml`
   - カリキュラム学習の設定が含まれています

### 質問・相談事項

実装中に不明点があれば、以下を確認してください:

1. **カバレッジ損失の計算方法**
   - implementation_guide.md の `calculate_coverage_loss()` を参照

2. **報酬計算の分岐ロジック**
   - implementation_guide.md の `_calculate_reward()` を参照

3. **カリキュラム学習の設定**
   - improved_reward_design.yaml を参照

---

## 📊 評価基準

### 各フェーズの合格基準

**Phase 1 の合格基準**:
- [ ] カバレッジ損失が計算できる
- [ ] 軽症系の報酬にカバレッジが反映される
- [ ] 学習が安定している（actor_loss, critic_lossが発散しない）
- [ ] 重症系の性能が悪化していない

**Phase 2 の合格基準**:
- [ ] 重症系平均RT < 10.87分
- [ ] 重症系6分達成率 > 20.8%
- [ ] 全体の13分達成率 > 50%
- [ ] カリキュラム学習が機能している

**Phase 3 の合格基準**:
- [ ] 重症系平均RT ≈ 10.58分（傷病度考慮運用レベル）
- [ ] 重症系6分達成率 ≈ 24.1%
- [ ] 計算時間が許容範囲（1エピソード < 10分）

---

## 🎉 成功の定義

**最低限の成功**:
- 直近隊運用に勝つ（重症系の平均RTと6分達成率）

**理想的な成功**:
- 傷病度考慮運用に匹敵する性能
- 学習が安定している
- 実用的な計算時間

**完全な成功**:
- 傷病度考慮運用を上回る性能
- 直近隊運用より全ての指標で優れている
- 実用化可能なレベル

---

**さあ、Phase 1 の実装を始めましょう！**
