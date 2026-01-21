# PPO環境再設計 設計仕様書

## 概要

本仕様書は、救急隊配車最適化のPPO学習環境を再設計するための詳細な設計と実装指示を記載します。

### 変更の目的

1. **状態空間の整理**: 46次元のコンパクトな状態空間に統一
2. **報酬設計の簡素化**: 5つのモードから1つの統一報酬関数に集約
3. **不要コードの削除**: 未使用のモード、パラメータ、関数を削除
4. **傷病度考慮運用のロジック移植**: 実績ある配車ロジックのカバレッジ計算をPPOに統合

---

## 1. 変更対象ファイルと変更内容の概要

| ファイル | 変更内容 |
|---------|---------|
| `state_encoder.py` | 状態空間の再設計（46次元）、カバレッジ損失計算の追加 |
| `reward_designer.py` | 報酬関数の簡素化、不要モードの削除 |
| `config.yaml` | 不要パラメータの削除、新パラメータの追加 |
| `ems_environment.py` | 状態エンコーダーとの連携修正（必要に応じて） |
| `trainer.py` | 不要なハイブリッド関連コードの整理 |

---

## 2. 状態空間の詳細定義（46次元）

### 2.1 全体構成

```
【候補隊情報】Top-10 × 4次元 = 40次元
  - 移動時間 (/ 30分)
  - 移動距離 (/ 10km)  
  - カバレッジ損失 L6 (0-1)
  - カバレッジ損失 L13 (0-1)

【グローバル状態】5次元
  - 利用可能率 (available / 192)
  - 出場中率 (dispatched / 192)
  - 6分圏内台数 (/ K)
  - 平均移動時間 (/ 30分)
  - システムカバレッジ C6

【事案情報】1次元
  - 傷病度 (0=重症系, 1=軽症系)

合計: 46次元
```

### 2.2 候補隊情報の実装

```python
# 各候補隊の特徴量（Top-K × 4次元 = 40次元）
for i, ambulance in enumerate(top_k_ambulances):
    base_idx = i * 4
    
    # 移動時間（正規化: /30分）
    features[base_idx + 0] = min(ambulance['travel_time_minutes'] / 30.0, 1.0)
    
    # 移動距離（正規化: /10km）
    features[base_idx + 1] = min(ambulance['travel_distance_km'] / 10.0, 1.0)
    
    # カバレッジ損失 L6（0-1）
    features[base_idx + 2] = ambulance['coverage_loss_6min']
    
    # カバレッジ損失 L13（0-1）
    features[base_idx + 3] = ambulance['coverage_loss_13min']
```

### 2.3 カバレッジ損失の計算（dispatch_strategies.pyから移植）

**重要**: この関数は `dispatch_strategies.py` の `SeverityBasedStrategy._calculate_coverage_loss` を参考に実装してください。

```python
def calculate_coverage_loss(ambulance, all_available, travel_time_matrix, grid_mapping):
    """
    救急隊が出場した場合のカバレッジ損失を計算
    傷病度考慮運用（SeverityBasedStrategy）と同じロジック
    
    Args:
        ambulance: 評価対象の救急隊（dict with 'id', 'current_h3', 'station_h3'）
        all_available: 利用可能な全救急隊のリスト
        travel_time_matrix: 移動時間行列（numpy array）
        grid_mapping: H3インデックス→行列インデックスのマッピング（dict）
    
    Returns:
        L6: 6分カバレッジ損失 (0-1)
        L13: 13分カバレッジ損失 (0-1)
    """
    import h3
    import numpy as np
    
    # この救急隊を除いた利用可能な救急隊リスト
    remaining = [amb for amb in all_available if amb['id'] != ambulance['id']]
    
    if not remaining:
        return 1.0, 1.0  # 他に救急隊がない場合は最大損失
    
    # サンプルポイントの取得（署所周辺、リング距離r=2以内）
    try:
        center_h3 = ambulance['station_h3']
        nearby_grids = h3.grid_disk(center_h3, 2)  # リング半径=2（傷病度考慮運用と同じ）
        sample_points = [g for g in nearby_grids if g in grid_mapping]
        
        # サンプルサイズを20に制限
        if len(sample_points) > 20:
            import random
            sample_points = random.sample(sample_points, 20)
    except Exception:
        return 0.5, 0.5  # エラー時はデフォルト
    
    if not sample_points:
        return 0.5, 0.5
    
    # 出場前後のカバレッジを計算
    coverage_6min_before = 0
    coverage_13min_before = 0
    coverage_6min_after = 0
    coverage_13min_after = 0
    
    TIME_THRESHOLD_6MIN = 360  # 秒
    TIME_THRESHOLD_13MIN = 780  # 秒
    
    for point_h3 in sample_points:
        point_idx = grid_mapping.get(point_h3)
        if point_idx is None:
            continue
        
        # 出場前：全救急隊での最小応答時間
        min_time_before = float('inf')
        for amb in all_available:
            amb_idx = grid_mapping.get(amb.get('current_h3'))
            if amb_idx is not None:
                try:
                    travel_time = travel_time_matrix[amb_idx, point_idx]
                    min_time_before = min(min_time_before, travel_time)
                except IndexError:
                    continue
        
        # 出場後：この救急隊を除いた最小応答時間
        min_time_after = float('inf')
        for amb in remaining:
            amb_idx = grid_mapping.get(amb.get('current_h3'))
            if amb_idx is not None:
                try:
                    travel_time = travel_time_matrix[amb_idx, point_idx]
                    min_time_after = min(min_time_after, travel_time)
                except IndexError:
                    continue
        
        # カバレッジのカウント
        if min_time_before <= TIME_THRESHOLD_6MIN:
            coverage_6min_before += 1
        if min_time_before <= TIME_THRESHOLD_13MIN:
            coverage_13min_before += 1
        if min_time_after <= TIME_THRESHOLD_6MIN:
            coverage_6min_after += 1
        if min_time_after <= TIME_THRESHOLD_13MIN:
            coverage_13min_after += 1
    
    # カバレッジ損失を計算
    total_points = len(sample_points)
    if total_points == 0:
        return 0.5, 0.5
    
    L6 = (coverage_6min_before - coverage_6min_after) / total_points
    L13 = (coverage_13min_before - coverage_13min_after) / total_points
    
    # 0-1の範囲にクリップ
    L6 = max(0.0, min(1.0, L6))
    L13 = max(0.0, min(1.0, L13))
    
    return L6, L13
```

### 2.4 グローバル状態（5次元）

```python
# グローバル状態のインデックス
global_base_idx = 40  # Top-K情報（40次元）の後

# 1. 利用可能率
available_count = sum(1 for a in ambulances.values() if a.get('status') == 'available')
features[global_base_idx + 0] = available_count / 192

# 2. 出場中率（利用可能率の補完）
dispatched_count = sum(1 for a in ambulances.values() if a.get('status') == 'dispatched')
features[global_base_idx + 1] = dispatched_count / 192

# 3. 6分圏内台数（Top-K内）
within_6min_count = sum(1 for a in top_k_list if a['travel_time_minutes'] <= 6)
features[global_base_idx + 2] = within_6min_count / self.top_k

# 4. 平均移動時間（Top-K）
valid_times = [a['travel_time_minutes'] for a in top_k_list if a.get('amb_id', -1) >= 0]
if valid_times:
    avg_travel_time = np.mean(valid_times)
else:
    avg_travel_time = 30.0  # デフォルト最大値
features[global_base_idx + 3] = min(avg_travel_time / 30.0, 1.0)

# 5. システムカバレッジ C6（既存の_calculate_coverage_rateを使用、閾値を6分に設定）
features[global_base_idx + 4] = self._calculate_coverage_rate_6min(ambulances, grid_mapping)
```

### 2.5 事案情報（1次元）

```python
# 傷病度（binary）
# constants.pyのis_severe_conditionを使用
from constants import is_severe_condition

severity_idx = 45
features[severity_idx] = 0.0 if is_severe_condition(severity) else 1.0
```

---

## 3. 報酬設計の詳細定義

### 3.1 設計方針

- **論文との整合性**: 時間報酬は論文5章の設計（重症系:指数/べき乗、軽症系:線形）を維持
- **カバレッジ報酬の追加**: 行動レベルのカバレッジ損失（L6, L13）に基づく報酬を新規追加
- **傷病度考慮運用との整合性**: 重み配分（time:coverage = 0.6:0.4）を採用

### 3.2 報酬関数の実装

```python
"""
reward_designer.py
簡素化された報酬設計システム
"""

import numpy as np
from typing import Dict
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from constants import is_severe_condition


class RewardDesigner:
    """
    簡素化された報酬設計クラス
    
    モード:
    - 通常PPO: 重症系・軽症系ともにPPO学習対象
    - ハイブリッドPPO: 重症系は直近隊運用（報酬なし）、軽症系のみPPO学習
    """
    
    def __init__(self, config: Dict):
        """
        報酬設計の初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        
        # ===== 報酬パラメータの読み込み =====
        reward_config = config.get('reward', {}).get('unified', {})
        
        # 重症系パラメータ（通常PPO用、論文5章の設計）
        self.critical_params = {
            'target': 6.0,  # 目標時間（分）
            'max_bonus': reward_config.get('critical_max_bonus', 50.0),
            'lambda_param': reward_config.get('critical_lambda', 0.115),
            'penalty_scale': reward_config.get('critical_penalty_scale', 5.0),
            'penalty_power': reward_config.get('critical_penalty_power', 1.5),
        }
        
        # 軽症系パラメータ（論文5章の設計）
        self.mild_params = {
            'target': 13.0,  # 目標時間（分）
            'max_bonus': reward_config.get('mild_max_bonus', 10.0),
            'penalty_scale': reward_config.get('mild_penalty_scale', 1.0),
        }
        
        # カバレッジパラメータ（傷病度考慮運用のロジックに基づく）
        self.coverage_params = {
            'w6': reward_config.get('coverage_w6', 0.5),
            'w13': reward_config.get('coverage_w13', 0.5),
            'penalty_scale': reward_config.get('coverage_penalty_scale', 10.0),
        }
        
        # 重み配分（傷病度考慮運用と同じ: time=0.6, coverage=0.4）
        self.time_weight = reward_config.get('time_weight', 0.6)
        self.coverage_weight = reward_config.get('coverage_weight', 0.4)
        
        # モード設定
        self.hybrid_mode = config.get('hybrid_mode', {}).get('enabled', False)
        
        # システムレベル設定
        system_config = config.get('reward', {}).get('system', {})
        self.dispatch_failure_penalty = system_config.get('dispatch_failure', -1.0)
        self.no_available_penalty = system_config.get('no_available_ambulance', 0.0)
        
        # 初期化完了ログ
        print(f"RewardDesigner初期化完了:")
        print(f"  ハイブリッドモード: {'有効' if self.hybrid_mode else '無効'}")
        print(f"  重み配分: time={self.time_weight}, coverage={self.coverage_weight}")
        print(f"  カバレッジ配分: w6={self.coverage_params['w6']}, w13={self.coverage_params['w13']}")
    
    def calculate_step_reward(self, severity: str, response_time_sec: float,
                              L6: float = 0.0, L13: float = 0.0) -> float:
        """
        ステップ報酬を計算
        
        Args:
            severity: 傷病度（'重症', '重篤', '死亡', '中等症', '軽症'）
            response_time_sec: 応答時間（秒）
            L6: 6分カバレッジ損失 (0-1)、状態エンコーダーから取得
            L13: 13分カバレッジ損失 (0-1)、状態エンコーダーから取得
        
        Returns:
            報酬値（-100 ~ 100にクリップ）
        """
        rt_min = response_time_sec / 60.0
        
        # ========== 重症系 ==========
        if is_severe_condition(severity):
            if self.hybrid_mode:
                # ハイブリッドモード: 直近隊運用、学習対象外
                return 0.0
            else:
                # 通常PPO: 論文どおりの報酬計算
                return self._calculate_critical_reward(rt_min)
        
        # ========== 軽症系 ==========
        # 時間報酬（論文どおり線形）
        time_reward = self._calculate_mild_time_reward(rt_min)
        
        # カバレッジ報酬（行動レベル、傷病度考慮運用のロジック）
        coverage_reward = self._calculate_coverage_reward(L6, L13)
        
        # 重み付け合計
        total_reward = (self.time_weight * time_reward + 
                       self.coverage_weight * coverage_reward)
        
        return np.clip(total_reward, -100.0, 100.0)
    
    def _calculate_critical_reward(self, rt_min: float) -> float:
        """
        重症系の報酬計算（通常PPO用）
        
        論文5章の設計:
        - 目標時間内: r = B_c × exp(-λt)
        - 目標超過時: r = -α_c × (t - T_c)^ν
        """
        p = self.critical_params
        
        if rt_min <= p['target']:
            # 指数関数的な報酬（早ければ早いほど高い報酬）
            return p['max_bonus'] * np.exp(-p['lambda_param'] * rt_min)
        else:
            # べき乗ペナルティ（超過するほど急激にペナルティ増加）
            overtime = rt_min - p['target']
            return -p['penalty_scale'] * (overtime ** p['penalty_power'])
    
    def _calculate_mild_time_reward(self, rt_min: float) -> float:
        """
        軽症系の時間報酬計算
        
        論文5章の設計:
        - 目標時間内: r = B_m × (1 - t/T_m)
        - 目標超過時: r = -α_m × (t - T_m)
        """
        p = self.mild_params
        
        if rt_min <= p['target']:
            # 線形報酬（目標時間に対する達成度）
            return p['max_bonus'] * (1 - rt_min / p['target'])
        else:
            # 線形ペナルティ
            overtime = rt_min - p['target']
            return -p['penalty_scale'] * overtime
    
    def _calculate_coverage_reward(self, L6: float, L13: float) -> float:
        """
        カバレッジ報酬計算（行動レベル）
        
        傷病度考慮運用のロジック:
        - 選んだ隊が出場することによるカバレッジ損失をペナルティとして計算
        - 損失が小さい = 良い選択 → 報酬が高い（ペナルティが小さい）
        - 損失が大きい = 悪い選択 → 報酬が低い（ペナルティが大きい）
        """
        p = self.coverage_params
        
        # カバレッジ損失の重み付け合計
        combined_loss = p['w6'] * L6 + p['w13'] * L13
        
        # 損失をペナルティに変換（損失が大きいほどペナルティが大きい）
        return -p['penalty_scale'] * combined_loss
    
    def get_failure_penalty(self, failure_type: str) -> float:
        """
        失敗ペナルティを取得
        
        Args:
            failure_type: 'dispatch', 'no_available', 'unhandled'
        """
        if failure_type == 'dispatch':
            return self.dispatch_failure_penalty
        elif failure_type == 'no_available':
            return self.no_available_penalty
        else:
            return -10.0
    
    def get_info(self) -> Dict:
        """現在の報酬設定情報を返す"""
        return {
            'hybrid_mode': self.hybrid_mode,
            'time_weight': self.time_weight,
            'coverage_weight': self.coverage_weight,
            'critical_params': self.critical_params,
            'mild_params': self.mild_params,
            'coverage_params': self.coverage_params,
        }
```

### 3.3 configの設定

```yaml
# config.yaml または個別設定ファイル

# ===== 統一報酬設定 =====
reward:
  # システムレベル（既存維持）
  system:
    dispatch_failure: -1.0
    no_available_ambulance: 0.0
  
  # 統一報酬パラメータ
  unified:
    # 重症系パラメータ（通常PPO用、論文5章の設計）
    critical_max_bonus: 50.0
    critical_lambda: 0.115
    critical_penalty_scale: 5.0
    critical_penalty_power: 1.5
    
    # 軽症系パラメータ（論文5章の設計）
    mild_max_bonus: 10.0
    mild_penalty_scale: 1.0
    
    # カバレッジパラメータ
    coverage_w6: 0.5       # L6の重み
    coverage_w13: 0.5      # L13の重み
    coverage_penalty_scale: 10.0  # カバレッジ損失のペナルティスケール
    
    # 重み配分（傷病度考慮運用と同じ）
    time_weight: 0.6
    coverage_weight: 0.4

# ===== ハイブリッドモード設定 =====
hybrid_mode:
  enabled: true  # true: ハイブリッドPPO, false: 通常PPO
  severity_classification:
    severe_conditions: ["重症", "重篤", "死亡"]
    mild_conditions: ["軽症", "中等症"]
```

---

## 4. ネットワーク構成の変更

### 4.1 変更内容

```yaml
network:
  actor:
    hidden_layers: [128, 64]  # 変更前: [256, 128]
    activation: relu
    dropout: 0.2              # 変更前: 0.1
    initialization: xavier_uniform
  
  critic:
    hidden_layers: [128, 64]  # 変更前: [256, 128]
    activation: relu
    dropout: 0.2              # 変更前: 0.1
    init_scale: 0.001
```

### 4.2 変更理由

- 入力次元（46次元）に対して適切なサイズ（約2.8倍）
- 過学習リスクの低減
- 報酬設計の効果を検証しやすくするため

---

## 5. 削除するコード

### 5.1 reward_designer.py

#### 削除する関数・メソッド

```python
# 完全削除
def _calculate_simple_reward(self, severity, response_time_minutes)
def _calculate_discrete_reward(self, severity, response_time_minutes)
def _calculate_coverage_aware_reward(self, severity, response_time_minutes, coverage_loss)
def _load_coverage_aware_params(self, override_config)
def _load_mode_params_from_reward_mode(self, reward_mode_config)
def _load_mode_params_from_core(self, core_config)

# 簡素化して残す
def _calculate_continuous_reward  # → _calculate_critical_reward, _calculate_mild_time_reward に分割
def _calculate_hybrid_reward      # → calculate_step_reward に統合
def _init_hybrid_mode             # 簡素化
```

#### 削除するクラス変数・パラメータ

```python
# 削除
self.mode  # モード選択は不要（hybrid_modeのみで制御）
self.simple_params
self.discrete_params
self.coverage_aware_params
self.continuous_params  # critical_params, mild_paramsに分割
self.hybrid_params      # unified設定に統合

# 削除
self.curriculum_enabled
self.curriculum_stages
self.current_episode
# （カリキュラム学習は一旦無効化、将来の拡張用にコメントアウトで残す）
```

#### 削除するモード分岐（calculate_step_reward内）

```python
# 削除する分岐
if self.mode == 'simple':
    reward = self._calculate_simple_reward(...)
elif self.mode == 'discrete':
    reward = self._calculate_discrete_reward(...)
elif self.mode == 'coverage_aware':
    reward = self._calculate_coverage_aware_reward(...)
```

### 5.2 config.yaml

#### 削除するセクション

```yaml
# 完全削除
reward:
  core:
    mode: ...              # 削除
    simple_params: {...}   # 削除
    discrete_params: {...} # 削除
    hybrid_params: {...}   # 削除（unifiedに統合）
    continuous_params: {...}  # 削除（unifiedに統合）
    coverage_impact_weight: ...  # 削除
  
  coverage_params: {...}   # 削除（unifiedに統合）
  episode: {...}           # 削除（エピソード報酬は使用しない）

reward_mode:               # 完全削除
  mode: ...
  simple: {...}
  continuous: {...}
  discrete: {...}
  coverage_aware: {...}

severity:                  # 簡素化（hybrid_mode.severity_classificationに統合）
  categories: {...}
  thresholds: {...}

teacher:                   # 削除（教師あり学習は使用しない）
  enabled: ...
  ...

curriculum:                # 削除（カリキュラム学習は一旦無効化）
  enabled: ...
  stages: [...]
```

### 5.3 trainer.py

#### 削除・統合するコード

```python
# 削除（hybrid_modeに統合）
self.hybrid_v2_mode = config.get('hybrid_v2', {}).get('enabled', False)
self.hybrid_v2_stats = {...}

# 削除（hybrid関連に統合）
def _update_hybrid_v2_stats(self, episode_stats)
def _log_hybrid_v2_metrics(self, episode_stats)

# 簡素化して残す
self.hybrid_mode = config.get('hybrid_mode', {}).get('enabled', False)
self.hybrid_stats = {...}  # 必要最小限に簡素化
def _update_hybrid_stats(self, episode_stats)  # 簡素化
def _log_hybrid_metrics(self, episode_stats)   # 簡素化
```

### 5.4 state_encoder.py

#### 削除・修正するコード

```python
# 削除（使用しない）
- coverage_aware_sorting の複雑なロジック（簡素化）
- 999次元エンコーダー関連のコード（使用しない場合）

# 修正
- _calculate_coverage_rate → _calculate_coverage_rate_6min（6分閾値に固定）
- カバレッジ損失計算を新規追加（calculate_coverage_loss関数）
```

---

## 6. テスト方法

### 6.1 状態空間のテスト

```python
def test_state_encoding():
    """状態エンコーディングのテスト"""
    from reinforcement_learning.environment.state_encoder import CompactStateEncoder
    
    # テスト用設定
    config = {
        'state_encoding': {
            'mode': 'compact',
            'top_k': 10,
            'normalization': {
                'max_travel_time_minutes': 30,
                'max_station_distance_km': 10
            }
        }
    }
    
    encoder = CompactStateEncoder(config)
    
    # ダミーデータで状態を生成
    # （実際のテストでは適切なダミーデータを用意）
    state = encoder.encode(...)
    
    # 1. 次元数の確認
    assert state.shape[0] == 46, f"Expected 46 dimensions, got {state.shape[0]}"
    
    # 2. 値の範囲確認（0-1に正規化されているか）
    assert np.all(state >= 0) and np.all(state <= 1), "State values out of range [0, 1]"
    
    # 3. カバレッジ損失の確認（各候補隊のL6, L13）
    for i in range(10):
        L6 = state[i * 4 + 2]
        L13 = state[i * 4 + 3]
        assert 0 <= L6 <= 1, f"L6 out of range: {L6}"
        assert 0 <= L13 <= 1, f"L13 out of range: {L13}"
    
    # 4. グローバル状態の確認
    available_rate = state[40]
    dispatched_rate = state[41]
    assert available_rate + dispatched_rate <= 1.0, "Available + Dispatched should be <= 1"
    
    print("✓ 状態エンコーディングテスト通過")
```

### 6.2 報酬計算のテスト

```python
def test_reward_calculation():
    """報酬計算のテスト"""
    from reinforcement_learning.rewards.reward_designer import RewardDesigner
    
    # テスト用設定
    config = {
        'reward': {
            'unified': {
                'critical_max_bonus': 50.0,
                'mild_max_bonus': 10.0,
                'mild_penalty_scale': 1.0,
                'coverage_w6': 0.5,
                'coverage_w13': 0.5,
                'coverage_penalty_scale': 10.0,
                'time_weight': 0.6,
                'coverage_weight': 0.4,
            }
        },
        'hybrid_mode': {'enabled': False}
    }
    
    reward_designer = RewardDesigner(config)
    
    # テスト1: 軽症系、目標時間内、カバレッジ損失小
    reward1 = reward_designer.calculate_step_reward(
        severity='軽症',
        response_time_sec=600,  # 10分
        L6=0.1,
        L13=0.05
    )
    print(f"テスト1（軽症、10分、L6=0.1, L13=0.05）: 報酬={reward1:.2f}")
    assert reward1 > 0, "目標時間内・カバレッジ損失小で報酬が正であるべき"
    
    # テスト2: 軽症系、目標時間超過
    reward2 = reward_designer.calculate_step_reward(
        severity='軽症',
        response_time_sec=1200,  # 20分
        L6=0.1,
        L13=0.05
    )
    print(f"テスト2（軽症、20分、L6=0.1, L13=0.05）: 報酬={reward2:.2f}")
    assert reward2 < reward1, "超過時は報酬が減少すべき"
    
    # テスト3: 軽症系、カバレッジ損失大
    reward3 = reward_designer.calculate_step_reward(
        severity='軽症',
        response_time_sec=600,  # 10分
        L6=0.8,
        L13=0.7
    )
    print(f"テスト3（軽症、10分、L6=0.8, L13=0.7）: 報酬={reward3:.2f}")
    assert reward3 < reward1, "カバレッジ損失大で報酬が減少すべき"
    
    # テスト4: 重症系（通常PPOモード）
    reward4 = reward_designer.calculate_step_reward(
        severity='重症',
        response_time_sec=300,  # 5分
        L6=0.1,
        L13=0.05
    )
    print(f"テスト4（重症、5分、通常PPO）: 報酬={reward4:.2f}")
    assert reward4 > 0, "重症系・目標時間内で報酬が正であるべき"
    
    # テスト5: 重症系（ハイブリッドモード）
    config['hybrid_mode']['enabled'] = True
    reward_designer_hybrid = RewardDesigner(config)
    reward5 = reward_designer_hybrid.calculate_step_reward(
        severity='重症',
        response_time_sec=300,  # 5分
        L6=0.1,
        L13=0.05
    )
    print(f"テスト5（重症、5分、ハイブリッドPPO）: 報酬={reward5:.2f}")
    assert reward5 == 0.0, "ハイブリッドモードの重症系は報酬0"
    
    print("✓ 報酬計算テスト通過")
```

### 6.3 統合テスト

```python
def test_integration():
    """統合テスト：1エピソード実行"""
    from reinforcement_learning.environment.ems_environment import EMSEnvironment
    from reinforcement_learning.agents.ppo_agent import PPOAgent
    
    # 設定読み込み
    config = load_config(...)
    
    env = EMSEnvironment(config)
    agent = PPOAgent(config)
    
    state = env.reset()
    total_reward = 0
    step_count = 0
    
    print("統合テスト開始...")
    
    for _ in range(100):  # 100ステップ
        # 状態の次元確認
        assert state.shape[0] == 46, f"State dim error: {state.shape[0]}"
        
        action = agent.select_action(state)
        result = env.step(action)
        
        total_reward += result.reward
        state = result.next_state
        step_count += 1
        
        if result.done:
            break
    
    print(f"✓ 統合テスト通過")
    print(f"  ステップ数: {step_count}")
    print(f"  総報酬: {total_reward:.2f}")
    print(f"  平均報酬: {total_reward/step_count:.2f}")
```

---

## 7. 実装の優先順位

1. **state_encoder.py**: 状態空間の再設計（カバレッジ損失計算の追加）
2. **reward_designer.py**: 報酬関数の簡素化（不要コード削除、新報酬関数実装）
3. **config.yaml**: 設定ファイルの整理（不要パラメータ削除、新パラメータ追加）
4. **trainer.py**: 不要コードの削除・統合
5. **ems_environment.py**: 必要に応じて連携修正
6. **テスト実行**: 各コンポーネントのテスト → 統合テスト

---

## 8. 注意事項

### 8.1 後方互換性

- 既存の学習済みモデルは使用できなくなります（状態空間が変更されるため）
- 新しい設定ファイル形式に移行が必要です

### 8.2 カバレッジ損失計算のパフォーマンス

- `calculate_coverage_loss`は各候補隊に対して呼び出されるため、計算コストに注意
- Top-K（10隊）×サンプルポイント（20点）= 200回の移動時間参照が発生
- 必要に応じてキャッシングを検討

### 8.3 削除するコードの保存

- 完全削除前に、関連コードをバックアップまたはコメントアウトで保存
- 将来の参照用に`_deprecated`ディレクトリへの移動も検討

---

---

## 9. network_architectures.py の修正

### 9.1 問題点

現在の実装では`ModularStateEncoder`がネットワーク内で状態を再エンコードしています。
新しい46次元設計では`CompactStateEncoder`が状態を出力するため、`ModularStateEncoder`は不要です。

### 9.2 修正内容

#### ActorNetwork の修正

```python
class ActorNetwork(nn.Module):
    """
    Actor Network（46次元状態空間対応版）
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # ★★★ ModularStateEncoderのデフォルトをFalseに変更 ★★★
        self.use_modular_encoder = config.get('use_modular_encoder', False)
        
        if self.use_modular_encoder:
            # 従来モード（999次元）用：レガシーサポート
            from environment.modular_state_encoder import ModularStateEncoder
            num_ambulances = config.get('num_ambulances', 192)
            self.state_encoder = ModularStateEncoder(max_ambulances=num_ambulances)
            encoded_dim = self.state_encoder.output_dim
        else:
            # 新設計（46次元）：状態をそのまま入力
            encoded_dim = state_dim
            self.state_encoder = None
        
        # ポリシーネットワーク構築
        hidden_layers = config.get('network', {}).get('actor', {}).get('hidden_layers', [128, 64])
        activation = config.get('network', {}).get('actor', {}).get('activation', 'relu')
        dropout_rate = config.get('network', {}).get('actor', {}).get('dropout', 0.2)
        
        # 以下、既存コードを維持...
```

#### CriticNetwork の修正

```python
class CriticNetwork(nn.Module):
    """
    Critic Network（46次元状態空間対応版）
    """
    
    def __init__(self, state_dim: int, config: Dict):
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # ★★★ ModularStateEncoderのデフォルトをFalseに変更 ★★★
        self.use_modular_encoder = config.get('use_modular_encoder', False)
        
        if self.use_modular_encoder:
            # 従来モード用
            from environment.modular_state_encoder import ModularStateEncoder
            num_ambulances = config.get('num_ambulances', 192)
            self.state_encoder = ModularStateEncoder(max_ambulances=num_ambulances)
            encoded_dim = self.state_encoder.output_dim
        else:
            # 新設計：状態をそのまま入力
            encoded_dim = state_dim
            self.state_encoder = None
        
        # 以下、既存コードを維持...
```

### 9.3 ppo_agent.py の確認

ppo_agent.pyは既に`use_modular_encoder`のデフォルトがFalseになっているため、**修正不要**です。

```python
# ppo_agent.py 52行目（既存）
'use_modular_encoder': config.get('use_modular_encoder', False),  # OK
```

### 9.4 config設定での制御

`use_modular_encoder`を明示的に設定することで、従来モード（999次元）と新モード（46次元）を切り替え可能です。

```yaml
# 新設計（46次元）- デフォルト
# use_modular_encoderは設定しない（False）

# 従来モード（999次元）- レガシーサポート
use_modular_encoder: true
```

---

## 10. config設定ファイルの構成

### 10.1 ベース設定と個別設定の関係

```
config.yaml（ベース設定）
    │
    │  load_config_with_inheritance()
    │
    ↓
config_xxx.yaml（個別設定）で上書き
```

### 10.2 個別設定ファイルの例

#### config_hybrid_experiment.yaml

```yaml
# ベース設定を継承
inherits: ./config.yaml

# 実験識別
experiment:
  name: "hybrid_ppo_experiment_v1"
  description: "ハイブリッドPPO実験（カバレッジ重視）"
  seed: 2025

# ハイブリッドモードを有効化
hybrid_mode:
  enabled: true
  severity_classification:
    severe_conditions: ["重症", "重篤", "死亡"]
    mild_conditions: ["軽症", "中等症"]

# 報酬パラメータの調整
reward:
  unified:
    # カバレッジをより重視
    time_weight: 0.5
    coverage_weight: 0.5
    coverage_penalty_scale: 15.0

# 訓練期間の設定
data:
  train_periods:
    - start_date: "20230401"
      end_date: "20230430"
  eval_periods:
    - start_date: "20230501"
      end_date: "20230507"

# PPOパラメータの調整
ppo:
  n_episodes: 2000
  batch_size: 512
```

#### config_normal_ppo.yaml

```yaml
# ベース設定を継承
inherits: ./config.yaml

# 実験識別
experiment:
  name: "normal_ppo_experiment_v1"
  description: "通常PPO実験（重症系も学習対象）"

# ハイブリッドモードは無効（デフォルト）
hybrid_mode:
  enabled: false

# 報酬パラメータ
reward:
  unified:
    # 重症系の報酬パラメータを調整
    critical_max_bonus: 60.0
    critical_penalty_scale: 8.0
```

---

## 変更履歴

| 日付 | 内容 |
|------|------|
| 2025-01-12 | 初版作成 |
| 2025-01-12 | network_architectures.pyの修正、config構成の説明を追加 |
