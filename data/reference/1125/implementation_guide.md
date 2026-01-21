# ===============================================
# 改良版報酬設計の実装ガイド
# ===============================================

## 📊 現状分析

### 傷病度考慮運用の成功要因

```
直近隊運用:
  - 重症系平均RT: 10.87分
  - 重症系6分達成率: 20.8%

傷病度考慮運用:
  - 重症系平均RT: 10.58分 ← 0.29分改善
  - 重症系6分達成率: 24.1% ← 3.3%改善
  
→ カバレッジを考慮した配車が有効であることを実証
```

### PPOが勝てない理由

1. **報酬設計が単純すぎる**
   ```python
   # 現在の報酬
   reward = -response_time × weight + bonus
   
   # 問題点:
   # - 応答時間のみを最小化 → 直近隊運用が最適解
   # - カバレッジの概念がない
   # - 将来の事案に備えた配車ができない
   ```

2. **傷病度考慮運用との違い**
   ```python
   # 傷病度考慮運用
   score = time_score × 0.6 + coverage_loss × 0.4
   
   # PPO（現在）
   reward = -time × 1.0 + 0 × coverage  # カバレッジが考慮されていない
   ```

---

## 🎯 改善方針

### 核心的なアイデア

**「傷病度考慮運用のロジックを報酬設計に組み込む」**

```
重症系:
  応答時間のみを重視（直近隊運用と同じ）
  → PPOは既にこれができている

軽症系:
  応答時間 × 0.6 + カバレッジ損失 × 0.4
  → PPOはこれができていない ← ここを改善
```

---

## 🔧 実装方法

### ステップ1: カバレッジ損失の計算関数を実装

```python
# reinforcement_learning/environment/ems_environment.py

def calculate_coverage_loss(
    self,
    selected_ambulance_id: int,
    available_ambulances: List[int],
    request_h3: str
) -> float:
    """
    選択した救急車が出動した場合のカバレッジ損失を計算
    
    Args:
        selected_ambulance_id: 選択した救急車のID
        available_ambulances: 利用可能な救急車IDのリスト
        request_h3: 要請地点のH3インデックス
    
    Returns:
        float: カバレッジ損失（0-1の範囲）
    """
    # 残りの救急車リスト
    remaining_ambulances = [
        amb_id for amb_id in available_ambulances 
        if amb_id != selected_ambulance_id
    ]
    
    if not remaining_ambulances:
        return 1.0  # 他に救急車がない場合は最大損失
    
    # 選択した救急車の現在位置（ステーション）
    selected_ambulance = self.ambulances[selected_ambulance_id]
    station_h3 = selected_ambulance.station_h3
    
    # 周辺グリッドをサンプリング（H3 ring 2以内）
    sample_points = self._get_coverage_sample_points(station_h3, sample_size=20)
    
    if not sample_points:
        # サンプルポイントが取得できない場合は簡易計算
        return self._simple_coverage_loss(
            station_h3, remaining_ambulances
        )
    
    # カバレッジ率を計算
    coverage_6min_before = 0
    coverage_13min_before = 0
    coverage_6min_after = 0
    coverage_13min_after = 0
    
    for point_h3 in sample_points:
        # 現在の状態でのカバレッジ
        min_time_before = self._get_min_response_time(
            point_h3, available_ambulances
        )
        if min_time_before <= 360:  # 6分
            coverage_6min_before += 1
        if min_time_before <= 780:  # 13分
            coverage_13min_before += 1
        
        # 救急車が出動した後のカバレッジ
        min_time_after = self._get_min_response_time(
            point_h3, remaining_ambulances
        )
        if min_time_after <= 360:
            coverage_6min_after += 1
        if min_time_after <= 780:
            coverage_13min_after += 1
    
    # カバレッジ損失を計算
    total_points = len(sample_points)
    loss_6min = (coverage_6min_before - coverage_6min_after) / total_points
    loss_13min = (coverage_13min_before - coverage_13min_after) / total_points
    
    # 重み付け合成（傷病度考慮運用と同じ）
    combined_loss = loss_6min * 0.5 + loss_13min * 0.5
    
    # 0-1の範囲にクリップ
    return max(0.0, min(1.0, combined_loss))


def _get_coverage_sample_points(
    self, 
    center_h3: str, 
    sample_size: int = 20
) -> List[str]:
    """カバレッジ計算用のサンプルポイントを取得"""
    try:
        import h3
        # 中心から2リング以内のグリッドを取得
        nearby_grids = h3.grid_disk(center_h3, 2)
        
        # grid_mappingに存在するグリッドのみを使用
        valid_grids = [
            g for g in nearby_grids 
            if g in self.grid_mapping
        ]
        
        # サンプルサイズを調整
        if len(valid_grids) <= sample_size:
            return valid_grids
        
        # ランダムサンプリング
        import random
        return random.sample(valid_grids, sample_size)
        
    except Exception as e:
        # エラーの場合は空リストを返す
        return []


def _get_min_response_time(
    self, 
    target_h3: str, 
    ambulance_ids: List[int]
) -> float:
    """指定地点への最小応答時間を取得"""
    if not ambulance_ids:
        return float('inf')
    
    min_time = float('inf')
    for amb_id in ambulance_ids:
        ambulance = self.ambulances[amb_id]
        travel_time = self.travel_time_estimator.estimate_travel_time(
            ambulance.current_h3, 
            target_h3, 
            'response'
        )
        if travel_time < min_time:
            min_time = travel_time
    
    return min_time


def _simple_coverage_loss(
    self, 
    station_h3: str, 
    remaining_ambulances: List[int]
) -> float:
    """簡易的なカバレッジ損失計算（近隣救急車数ベース）"""
    nearby_count = 0
    threshold_time = 600  # 10分
    
    for amb_id in remaining_ambulances:
        ambulance = self.ambulances[amb_id]
        travel_time = self.travel_time_estimator.estimate_travel_time(
            ambulance.current_h3, 
            station_h3, 
            'response'
        )
        if travel_time <= threshold_time:
            nearby_count += 1
    
    # 近隣救急車が多いほど損失は小さい
    return 1.0 / (nearby_count + 1)
```

### ステップ2: 報酬計算ロジックの修正

```python
# reinforcement_learning/environment/ems_environment.py
# step()メソッド内の報酬計算部分

def _calculate_reward(
    self, 
    request: EmergencyRequest, 
    selected_ambulance_id: int,
    response_time: float,
    available_ambulances: List[int]
) -> float:
    """報酬を計算（カバレッジ考慮版）"""
    
    severity = request.severity
    rt_minutes = response_time / 60.0
    
    # ===== 重症系の報酬 =====
    if severity in ['重症', '重篤', '死亡']:
        # 応答時間ベースの報酬（従来通り）
        time_component = -rt_minutes * 3.0  # weight=3.0
        
        # 6分ボーナス・ペナルティ
        if rt_minutes <= 6:
            bonus = 100.0
        else:
            bonus = -(rt_minutes - 6) * 10.0
        
        reward = time_component + bonus
        
        return reward
    
    # ===== 軽症系の報酬 =====
    else:  # 軽症、中等症
        # 1. 応答時間コンポーネント（60%）
        time_weight = 0.6
        
        if severity == '中等症':
            severity_weight = 1.5
        else:  # 軽症
            severity_weight = 0.5
        
        time_component = -rt_minutes * severity_weight * time_weight
        
        # 2. カバレッジコンポーネント（40%）
        coverage_weight = 0.4
        coverage_loss = self.calculate_coverage_loss(
            selected_ambulance_id,
            available_ambulances,
            request.h3_index
        )
        coverage_component = -coverage_loss * 100.0 * coverage_weight
        
        # 3. ボーナス・ペナルティ
        if rt_minutes <= 13:
            bonus = 30.0 if severity == '中等症' else 10.0
        elif rt_minutes <= 20:
            bonus = -(rt_minutes - 13) * 3.0
        else:
            bonus = -50.0
        
        # 合計
        reward = time_component + coverage_component + bonus
        
        return reward
```

### ステップ3: アクションマスクの強化

```python
# reinforcement_learning/environment/ems_environment.py

def _get_action_mask_with_coverage(
    self,
    request: EmergencyRequest,
    available_ambulances: List[int]
) -> np.ndarray:
    """カバレッジを考慮したアクションマスク"""
    
    mask = np.zeros(self.action_dim, dtype=bool)
    severity = request.severity
    
    # 軽症系の場合、時間制約とカバレッジ損失でフィルタ
    if severity not in ['重症', '重篤', '死亡']:
        time_limit = 780  # 13分
        coverage_threshold = 0.8  # 損失80%以上をマスク
        
        for amb_id in available_ambulances:
            # 応答時間をチェック
            ambulance = self.ambulances[amb_id]
            response_time = self.travel_time_estimator.estimate_travel_time(
                ambulance.current_h3,
                request.h3_index,
                'response'
            )
            
            # 13分以内 かつ カバレッジ損失が許容範囲
            if response_time <= time_limit:
                coverage_loss = self.calculate_coverage_loss(
                    amb_id,
                    available_ambulances,
                    request.h3_index
                )
                if coverage_loss < coverage_threshold:
                    mask[amb_id] = True
        
        # マスクされた選択肢がない場合は全て許可
        if not mask.any():
            for amb_id in available_ambulances:
                mask[amb_id] = True
    
    # 重症系の場合、全て許可
    else:
        for amb_id in available_ambulances:
            mask[amb_id] = True
    
    return mask
```

---

## 📈 期待される効果

### 定量的な目標

```
現状（ep3000-重症重視×アクティブ）:
  - 重症系平均RT: 10.98分
  - 重症系6分達成率: 19.5%

目標（改良版）:
  - 重症系平均RT: 10.5分以下（直近隊: 10.87分を上回る）
  - 重症系6分達成率: 22%以上（直近隊: 20.8%を上回る）
  
理想（傷病度考慮運用レベル）:
  - 重症系平均RT: 10.58分
  - 重症系6分達成率: 24.1%
```

### 学習の見通し

1. **Stage 1 (ep0-1000)**:
   - カバレッジなし、応答時間のみ学習
   - 直近隊運用に近い性能

2. **Stage 2 (ep1000-3000)**:
   - カバレッジを徐々に導入
   - 軽症系でカバレッジを考慮した配車を学習開始

3. **Stage 3 (ep3000-5000)**:
   - 最終的なバランス（time 60% + coverage 40%）
   - 傷病度考慮運用に近い性能に到達

---

## 🚀 実装の優先順位

### Phase 1: 最小限の実装（1-2日）

1. `calculate_coverage_loss()`関数の実装
2. 報酬計算ロジックの修正（重症系と軽症系で分岐）
3. 簡単なテスト実行（ep1000程度）

**目的**: カバレッジ損失を報酬に組み込む効果を確認

### Phase 2: 完全版の実装（2-3日）

1. アクションマスクの強化
2. カリキュラム学習の導入
3. 状態表現の拡張（オプション）
4. 本格的な学習（ep5000）

**目的**: 傷病度考慮運用レベルの性能を達成

### Phase 3: 最適化（1-2日）

1. ハイパーパラメータチューニング
2. カバレッジ損失計算の高速化
3. ネットワークアーキテクチャの調整

**目的**: さらなる性能向上

---

## 💡 重要なポイント

1. **傷病度考慮運用の成功パターンを模倣する**
   - 重症系: 応答時間のみ
   - 軽症系: 応答時間 + カバレッジ

2. **カバレッジ損失を定量化する**
   - 6分カバレッジと13分カバレッジの変化
   - 重み: 50% + 50%

3. **段階的に導入する**
   - まずは応答時間のみ学習
   - 徐々にカバレッジを導入
   - 最終的なバランスに到達

4. **計算コストに注意**
   - カバレッジ損失計算は重い（H3グリッドのサンプリング）
   - サンプルサイズを調整（20ポイント程度）
   - 必要に応じて事前計算・キャッシュを活用

---

## 📊 検証方法

### 学習中の確認事項

```python
# wandbでモニタリング
wandb.log({
    'reward/time_component': time_component,
    'reward/coverage_component': coverage_component,
    'reward/total': reward,
    'coverage/mean_loss': mean_coverage_loss,
    'coverage/6min_rate': coverage_6min_rate,
    'coverage/13min_rate': coverage_13min_rate,
})
```

### 評価時の確認事項

1. **重症系の性能**
   - 平均RT < 10.87分（直近隊）
   - 6分達成率 > 20.8%（直近隊）

2. **軽症系の性能**
   - カバレッジ損失が小さい配車ができているか
   - 平均RTが極端に悪化していないか

3. **全体のバランス**
   - 13分達成率が維持されているか
   - 全体の平均RTが許容範囲か

---

## 🎯 まとめ

**核心的な改善点**:
```
PPOの報酬設計に「カバレッジ損失」を組み込む

reward = -time × time_weight + (-coverage_loss) × coverage_weight + bonus
         ↑                      ↑
         従来通り                新規追加（傷病度考慮運用から学ぶ）
```

**これにより**:
- 軽症系でカバレッジを考慮した配車が可能
- 将来の事案に備えた最適化ができる
- 傷病度考慮運用と同等の性能が期待できる
