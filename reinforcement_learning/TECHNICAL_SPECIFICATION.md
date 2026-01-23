# 救急車配車最適化システム - PPO学習と比較実験 技術仕様書

## 1. システム概要

本システムは、東京都を対象とした救急車配車最適化のための強化学習システムです。Proximal Policy Optimization (PPO) アルゴリズムを用いて、救急車の配車戦略を学習します。

### 1.1 主要目標
- **重症系事案（重症・重篤・死亡）**: 6分以内の応答時間達成
- **軽症系事案（軽症・中等症）**: 13分以内の応答時間達成
- **カバレッジ維持**: 配車後も周辺地域のカバレッジを維持

### 1.2 動作モード
1. **通常PPO**: 全傷病度に対してPPOが配車を学習
2. **ハイブリッドPPO**: 重症系は直近隊運用（最寄り選択）、軽症系のみPPO学習

---

## 2. ディレクトリ構成

```
reinforcement_learning/
├── agents/                      # PPOエージェント関連
│   ├── ppo_agent.py            # PPOアルゴリズム実装
│   ├── buffer.py               # 経験リプレイバッファ
│   └── network_architectures.py # Actor-Criticネットワーク
├── environment/                 # 学習環境関連
│   ├── ems_environment.py      # OpenAI Gym形式の環境
│   ├── state_encoder.py        # 状態エンコーダー（Full/Compact）
│   ├── reward_designer.py      # 報酬設計
│   ├── modular_state_encoder.py # モジュラー状態エンコーダー
│   └── dispatch_logger.py      # 配車ログ
├── training/                    # 学習管理
│   └── trainer.py              # PPOトレーナー
├── experiments/                 # 実験設定・結果
│   ├── config_*.yaml           # 各種設定ファイル
│   └── ppo_training/           # 学習済みモデル・ログ
├── config_utils.py             # 設定ファイル処理
└── init.py
```

---

## 3. コアコンポーネント詳細

### 3.1 PPOエージェント (`agents/ppo_agent.py`)

#### クラス構成
```python
class PPOAgent:
    def __init__(self, state_dim, action_dim, config, device):
        """
        state_dim: 状態空間の次元（46次元 or 999次元等）
        action_dim: 行動空間の次元（救急車数、コンパクト時は10）
        config: PPO設定辞書
        device: 計算デバイス（cuda/cpu）
        """
```

#### 主要メソッド
- `select_action(state, action_mask, deterministic)`: 行動選択
- `update()`: PPO更新（GAEによるAdvantage計算含む）
- `store_transition(...)`: 経験をバッファに保存
- `save(path)` / `load(path)`: モデルの保存・読み込み

#### PPOハイパーパラメータ
| パラメータ | 典型値 | 説明 |
|-----------|--------|------|
| `clip_epsilon` | 0.1-0.2 | PPOクリッピング係数 |
| `gamma` | 0.99 | 割引率 |
| `gae_lambda` | 0.95 | GAE係数 |
| `n_epochs` | 4-10 | 更新エポック数 |
| `batch_size` | 64-512 | ミニバッチサイズ |
| `entropy_coef` | 0.01-0.05 | エントロピー正則化係数 |
| `learning_rate.actor` | 3e-4 | Actor学習率 |
| `learning_rate.critic` | 1e-3 | Critic学習率 |

### 3.2 ネットワークアーキテクチャ (`agents/network_architectures.py`)

#### ActorNetwork
```python
class ActorNetwork(nn.Module):
    """
    入力: 状態ベクトル [batch_size, state_dim]
    出力: 行動確率分布 [batch_size, action_dim]
    
    構成:
    - オプション: ModularStateEncoder（特徴量タイプ別エンコード）
    - 隠れ層: [128, 64] + LayerNorm + Dropout
    - Softmax出力
    """
```

#### CriticNetwork
```python
class CriticNetwork(nn.Module):
    """
    入力: 状態ベクトル [batch_size, state_dim]
    出力: 状態価値 [batch_size, 1]
    
    構成: ActorNetworkと同様、価値ヘッドのみ異なる
    """
```

### 3.3 経験バッファ (`agents/buffer.py`)

```python
class RolloutBuffer:
    """
    PPO用ロールアウトバッファ
    
    保存データ:
    - states: 状態ベクトル
    - actions: 選択された行動
    - rewards: 報酬
    - log_probs: 行動の対数確率
    - values: 状態価値推定
    - action_masks: 利用可能行動のマスク
    - dones: エピソード終了フラグ
    """
```

---

## 4. 環境システム (`environment/`)

### 4.1 EMSEnvironment (`ems_environment.py`)

OpenAI Gym互換の救急配車シミュレーション環境。

#### 初期化パラメータ
- `config_path`: YAML設定ファイルのパス
- `mode`: "train" または "eval"

#### 主要メソッド

**`reset(period_index=None) -> np.ndarray`**
- 環境をリセットし初期観測を返す
- 学習期間または評価期間からランダムに期間を選択
- 救急車状態を初期化、事案データを読み込み

**`step(action: int) -> StepResult`**
- 配車行動を実行
- コンパクトモード時: actionはTop-K内インデックス（0-9）
- 従来モード時: actionは救急車ID（0-191）
- ハイブリッドモード時: 重症系は直近隊を強制選択

**`get_action_mask() -> np.ndarray`**
- 利用可能な救急車のマスクを返す
- True = 選択可能、False = 選択不可

**`get_optimal_action() -> int`**
- 直近隊（最も応答時間が短い救急車）を返す
- 教師あり学習やベースライン比較に使用

#### StepResult構造
```python
@dataclass
class StepResult:
    observation: np.ndarray  # 次状態
    reward: float           # 報酬
    done: bool              # エピソード終了フラグ
    info: Dict[str, Any]    # 追加情報（応答時間、配車結果等）
```

### 4.2 状態エンコーダー (`state_encoder.py`)

#### StateEncoder（従来版）
約999次元の状態ベクトルを生成。

**状態ベクトル構成:**
```
[救急車特徴量: 192×5=960] + [事案特徴量: 10] + [時間特徴量: 8] + [空間特徴量: 21]
= 999次元
```

**救急車特徴量（5次元/台）:**
1. 緯度（正規化）
2. 経度（正規化）
3. 利用可能フラグ
4. 本日出動回数（正規化）
5. 事案現場までの移動時間（正規化）

#### CompactStateEncoder（46次元版）

最も近いTop-K救急車のみを考慮するコンパクトな設計。

**状態ベクトル構成:**
```
【候補隊情報】Top-10 × 4次元 = 40次元
  [i*4+0] 移動時間 (/ 30分)
  [i*4+1] 移動距離 (/ 10km)
  [i*4+2] カバレッジ損失 L6 (0-1)
  [i*4+3] カバレッジ損失 L13 (0-1)

【グローバル状態】5次元
  [40] 利用可能率 (available / 192)
  [41] 出場中率 (dispatched / 192)
  [42] 6分圏内台数 (/ K)
  [43] 平均移動時間 (/ 30分)
  [44] システムカバレッジ C6

【事案情報】1次元
  [45] 傷病度 (0=重症系, 1=軽症系)

合計: 46次元
```

**カバレッジ損失計算:**
```python
def calculate_coverage_loss(ambulance, all_available, travel_time_matrix, grid_mapping):
    """
    救急隊が出場した場合のカバレッジ損失を計算
    
    手順:
    1. 署所周辺のサンプルポイントを取得（リング距離r以内）
    2. 出場前後の6分/13分カバレッジを計算
    3. 損失 = (before - after) / total_points
    
    Returns:
        L6: 6分カバレッジ損失 (0-1)
        L13: 13分カバレッジ損失 (0-1)
    """
```

### 4.3 報酬設計 (`reward_designer.py`)

#### RewardDesigner

**モード:**
- 通常PPO: 重症系・軽症系ともに報酬を計算
- ハイブリッドPPO: 軽症系のみ報酬を計算（重症系は0）

**報酬計算式:**

**重症系（通常PPO時）:**
```
目標時間内(6分以内): r = B_c × exp(-λt)
  - B_c = 50.0（最大ボーナス）
  - λ = 0.115
  
目標超過時: r = -α_c × (t - T_c)^ν
  - α_c = 5.0
  - ν = 1.5
```

**軽症系:**
```python
total_reward = time_weight × time_reward + coverage_weight × coverage_reward

# 時間報酬
目標時間内(13分以内): time_reward = B_m × (1 - t/T_m)
  - B_m = 10.0
  
目標超過時: time_reward = -α_m × (t - T_m)
  - α_m = 1.0

# カバレッジ報酬
coverage_reward = -penalty_scale × (w6 × L6 + w13 × L13)
  - penalty_scale = 10.0~35.0
  - w6 = 0.5, w13 = 0.5
```

**デフォルト重み配分:**
- time_weight = 0.6
- coverage_weight = 0.4

---

## 5. 学習システム (`training/trainer.py`)

### PPOTrainer

#### 初期化
```python
trainer = PPOTrainer(agent, env, config, output_dir)
```

#### 学習ループ
```python
def train(self, start_episode=0):
    for episode in range(start_episode, n_episodes):
        # 1. エピソード実行
        reward, length, stats = self._run_episode(training=True)
        
        # 2. PPO更新（バッファが十分な場合）
        if len(agent.buffer) >= batch_size:
            update_stats = agent.update()
        
        # 3. 定期評価
        if episode % eval_interval == 0:
            eval_reward = self._evaluate()
            
        # 4. チェックポイント保存
        if episode % checkpoint_interval == 0:
            self._save_checkpoint(episode)
```

#### ハイブリッドモード処理
```python
def _run_episode(self, training=True):
    while not done:
        if hybrid_mode and is_severe_condition(severity):
            # 重症系: PPOの出力は使うが、直近隊が強制選択される
            action, log_prob, value = agent.select_action(state, mask)
            skip_learning = True  # 経験を保存しない
        else:
            # 軽症系: PPOが選択、経験を保存
            action, log_prob, value = agent.select_action(state, mask)
            skip_learning = False
```

---

## 6. 設定ファイル形式

### 6.1 基本構造

```yaml
# 実験メタ情報
experiment:
  name: "hybrid_unified_v8g"
  seed: 2025
  device: "cuda"

# PPO設定
ppo:
  n_episodes: 500
  batch_size: 512
  clip_epsilon: 0.1
  learning_rate:
    actor: 0.0003
    critic: 0.001
  gamma: 0.99
  gae_lambda: 0.95
  entropy_coef: 0.035

# 状態エンコーディング
state_encoding:
  mode: 'compact'  # 'compact' or 'full'
  top_k: 10
  coverage_aware_sorting:
    sample_radius: 4
    sample_size: 61

# ハイブリッドモード
hybrid_mode:
  enabled: true
  severity_classification:
    severe_conditions: ["重症", "重篤", "死亡"]
    mild_conditions: ["軽症", "中等症"]

# 報酬設定
reward:
  unified:
    mild_max_bonus: 10.0
    coverage_penalty_scale: 35.0
    time_weight: 0.6
    coverage_weight: 0.4

# データ設定
data:
  episode_duration_hours: 24
  train_periods:
    - start_date: "20230401"
      end_date: "20230430"
  eval_periods:
    - start_date: "20230501"
      end_date: "20230507"
```

### 6.2 設定継承

`inherits: ./config.yaml` を指定することで、ベース設定を継承し差分のみ上書き可能。

---

## 7. 比較実験システム (`baseline_comparison.py`)

### 7.1 概要

複数のディスパッチ戦略を同一条件で比較評価するシステム。

### 7.2 利用可能な戦略

| 戦略名 | クラス | 説明 |
|--------|--------|------|
| `closest` | ClosestAmbulanceStrategy | 移動時間最小の救急車を選択 |
| `closest_distance` | ClosestDistanceStrategy | 移動距離最小の救急車を選択 |
| `severity_based` | SeverityBasedStrategy | 傷病度に応じてカバレッジを考慮 |
| `second_ride` | SecondRideStrategy | 軽症系は2番目に近い救急車を選択 |
| `mexclp` | MEXCLPStrategy | 期待カバレッジ最大化 |
| `ppo_agent` | PPOStrategy | 学習済みPPOモデル |

### 7.3 実験実行

```python
run_comparison_experiment(
    start_date="20240204",
    end_date="20240210",
    episode_duration_hours=24,
    num_runs=100,  # ランダムサンプリング回数
    strategies=['closest', 'ppo_slot1', 'ppo_slot2']
)
```

### 7.4 評価指標

- 平均応答時間（全体/重症系/軽症系）
- 6分達成率（全体/重症系）
- 13分達成率
- 直近隊選択率
- 統計的有意性（t検定、ANOVA）

---

## 8. PPO戦略のテスト実行 (`dispatch_strategies.py`)

### 8.1 PPOStrategy クラス

学習済みPPOモデルを配車戦略として使用。

```python
class PPOStrategy(DispatchStrategy):
    def __init__(self):
        self.compact_mode = False  # True: 46次元, False: 999次元
        self.top_k = 10
        self.hybrid_mode = False
        
    def initialize(self, config):
        """
        config:
          - model_path: 学習済みモデルのパス
          - config_path: 設定ファイルのパス
          - hybrid_mode: ハイブリッドモード有効化
        """
        
    def select_ambulance(self, request, available_ambulances, travel_time_func, context):
        """
        1. ハイブリッドモード時、重症系は直近隊を返す
        2. 軽症系はPPOモデルで選択
        3. 状態をエンコード → PPO推論 → 行動をマッピング
        """
```

### 8.2 コンパクトモードの処理

```python
# 1. Top-K救急車を取得（移動時間順）
top_k_ambulances = self._get_top_k_ambulances(request, available_ambulances, travel_time_func)

# 2. 状態エンコード
state_dict['top_k_ambulances'] = top_k_info
state_vector = self.state_encoder.encode_state(state_dict)

# 3. PPO推論（action = 0~9のインデックス）
action = self.agent.select_action(state_vector, action_mask)

# 4. Top-Kリストから救急車を取得
selected_ambulance = top_k_ambulances[action]
```

---

## 9. 学習実行手順

### 9.1 学習開始

```bash
python train_ppo.py --experiment config_hybrid_unified_v8g.yaml
```

### 9.2 オプション

| オプション | 説明 |
|-----------|------|
| `--config` | 設定ファイルパス |
| `--experiment` | experiments/内の設定ファイル名 |
| `--experiment_name` | カスタム実験名 |
| `--resume` | チェックポイントから再開 |
| `--hybrid` | ハイブリッドモード有効化 |
| `--debug` | デバッグモード（10エピソード） |

### 9.3 出力ディレクトリ

```
reinforcement_learning/experiments/ppo_training/{experiment_name}/
├── checkpoints/
│   ├── best_model.pth
│   └── checkpoint_ep*.pth
├── logs/
│   └── training_stats.json
├── configs/
│   ├── config.yaml
│   └── config.json
└── visualizations/
```

---

## 10. モデル評価手順

### 10.1 単一戦略評価

```python
from validation_simulation import run_validation_simulation

run_validation_simulation(
    target_date_str="20240204",
    dispatch_strategy="ppo_slot1",
    strategy_config={
        'model_path': 'path/to/best_model.pth',
        'config_path': 'path/to/config.yaml',
        'hybrid_mode': True
    }
)
```

### 10.2 比較実験

```bash
python baseline_comparison.py
```

`EXPERIMENT_CONFIG['strategies']` で比較する戦略を指定。

---

## 11. 主要な設計決定

### 11.1 コンパクトモード導入理由

- **問題**: 192台全ての救急車を状態に含めると約999次元となり、学習が困難
- **解決**: Top-10の候補隊のみを考慮し46次元に圧縮
- **利点**: 
  - 学習が高速化
  - 行動空間も10次元に削減
  - カバレッジ情報を直接状態に含められる

### 11.2 ハイブリッドモード導入理由

- **問題**: 重症系の応答時間最適化と軽症系のカバレッジ維持は目標が異なる
- **解決**: 
  - 重症系: 直近隊運用（確実に最速対応）
  - 軽症系: PPO学習（カバレッジと時間のトレードオフを学習）
- **利点**: 重症系の性能を保証しつつ、軽症系で戦略的配車が可能

### 11.3 カバレッジ損失計算

- **目的**: 配車による周辺地域のカバレッジ低下を定量化
- **方法**: 
  - 署所周辺のサンプルポイントでカバレッジ変化を計算
  - L6（6分圏）とL13（13分圏）の両方を考慮
- **パラメータ調整**: 
  - `sample_radius`: 2（局所）→ 6（広域）で粒度調整
  - 値が大きいほど直近隊選択率が上昇

---

## 12. トラブルシューティング

### 12.1 学習が進まない

1. `entropy_coef` を増加（探索促進）
2. `learning_rate` を調整
3. `batch_size` を増加（安定化）

### 12.2 応答時間が悪化

1. `time_weight` を増加
2. `coverage_penalty_scale` を減少
3. ハイブリッドモードの場合、重症系は直近隊が選択されているか確認

### 12.3 テスト時の次元不一致

1. 学習時と同じ設定ファイルを使用しているか確認
2. `state_encoding.mode` が一致しているか確認
3. `compact_mode` 時は `top_k` 値が一致しているか確認

---

## 13. 参考文献

- Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- Jagtenberg et al., "Dynamic ambulance dispatching: is the closest-idle policy always optimal?", 2017
- 東京消防庁 救急活動データ

---

## 更新履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2026-01-23 | 1.0 | 初版作成 |
