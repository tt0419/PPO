# API リファレンス

本ドキュメントは、救急車配車最適化システムの主要APIを詳細に記述します。

---

## 1. PPOAgent API

### 1.1 初期化

```python
from reinforcement_learning.agents.ppo_agent import PPOAgent

agent = PPOAgent(
    state_dim=46,        # 状態空間次元
    action_dim=10,       # 行動空間次元
    config={
        'learning_rate': {'actor': 3e-4, 'critic': 1e-3},
        'clip_epsilon': 0.1,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'n_epochs': 4,
        'batch_size': 512,
        'entropy_coef': 0.035
    },
    device='cuda'  # or 'cpu'
)
```

### 1.2 行動選択

```python
action, log_prob, value = agent.select_action(
    state,              # np.ndarray: 状態ベクトル [state_dim]
    action_mask=mask,   # np.ndarray: 利用可能行動 [action_dim] (bool)
    deterministic=False # bool: True=最大確率選択, False=確率的サンプリング
)
# Returns:
#   action: int - 選択された行動インデックス
#   log_prob: float - 行動の対数確率
#   value: float - 状態価値推定
```

### 1.3 経験保存

```python
agent.store_transition(
    state=state,           # np.ndarray
    action=action,         # int
    reward=reward,         # float
    next_state=next_state, # np.ndarray
    done=done,             # bool
    log_prob=log_prob,     # float
    value=value,           # float
    action_mask=mask       # np.ndarray (optional)
)
```

### 1.4 更新

```python
update_stats = agent.update()
# Returns: Dict[str, float]
#   - 'actor_loss': Actor損失
#   - 'critic_loss': Critic損失
#   - 'entropy': エントロピー
#   - 'kl_divergence': KLダイバージェンス
```

### 1.5 保存・読み込み

```python
agent.save('path/to/model.pth')
agent.load('path/to/model.pth')
```

---

## 2. EMSEnvironment API

### 2.1 初期化

```python
from reinforcement_learning.environment.ems_environment import EMSEnvironment

env = EMSEnvironment(
    config_path='reinforcement_learning/experiments/config.yaml',
    mode='train'  # 'train' or 'eval'
)
```

### 2.2 リセット

```python
observation = env.reset(period_index=None)
# Args:
#   period_index: int (optional) - 特定期間を指定、Noneでランダム選択
# Returns:
#   observation: np.ndarray - 初期状態ベクトル
```

### 2.3 ステップ実行

```python
result = env.step(action)
# Args:
#   action: int - 選択する救急車のインデックス
#              コンパクトモード: 0-9 (Top-K内インデックス)
#              従来モード: 0-191 (救急車ID)
# Returns:
#   result: StepResult
#     - observation: np.ndarray - 次状態
#     - reward: float - 報酬
#     - done: bool - エピソード終了フラグ
#     - info: Dict - 追加情報
```

### 2.4 行動マスク取得

```python
action_mask = env.get_action_mask()
# Returns:
#   action_mask: np.ndarray[bool] - True=選択可能, False=選択不可
#   shape: [action_dim]
```

### 2.5 最適行動取得

```python
optimal_action = env.get_optimal_action()
# Returns:
#   optimal_action: int - 直近隊（最速到着）の行動インデックス
```

### 2.6 モード切替

```python
env.set_mode('eval')  # 'train' or 'eval'
```

### 2.7 プロパティ

```python
env.state_dim   # int: 状態空間次元
env.action_dim  # int: 行動空間次元（コンパクト時10、従来時192）
```

### 2.8 info辞書の内容

```python
info = {
    'dispatch_result': {
        'success': bool,
        'ambulance_id': int,
        'response_time': float,  # 秒
        'distance': float        # km
    },
    'dispatch_type': str,  # 'hybrid_v2_direct_closest' or 'hybrid_v2_ppo_filtered'
    'severity': str,
    'skipped_learning': bool,
    'coverage_info': {
        'overall_coverage': float,
        'L6': float,
        'L13': float
    }
}
```

---

## 3. StateEncoder API

### 3.1 StateEncoder（従来版）

```python
from reinforcement_learning.environment.state_encoder import StateEncoder

encoder = StateEncoder(
    config=config,
    max_ambulances=192,
    travel_time_matrix=np.load('travel_time.npy'),
    grid_mapping=grid_mapping_dict
)

state_vector = encoder.encode_state(state_dict)
# Args:
#   state_dict: Dict
#     - 'ambulances': Dict[int, Dict] - 救急車状態
#     - 'pending_call': Dict or None - 現在の事案
#     - 'episode_step': int
#     - 'time_of_day': float
# Returns:
#   state_vector: np.ndarray[float32] - shape [state_dim]

dim = encoder.state_dim  # int: 状態次元
```

### 3.2 CompactStateEncoder（46次元版）

```python
from reinforcement_learning.environment.state_encoder import CompactStateEncoder

encoder = CompactStateEncoder(
    config=config,
    top_k=10,
    travel_time_matrix=travel_time_matrix,
    grid_mapping=grid_mapping
)

state_vector = encoder.encode_state(state_dict)
# Returns: np.ndarray[float32] shape [46]

top_k_ids = encoder.get_top_k_ambulance_ids(ambulances, incident)
# Returns: List[int] - Top-K救急車のIDリスト
```

### 3.3 ファクトリ関数

```python
from reinforcement_learning.environment.state_encoder import create_state_encoder

encoder = create_state_encoder(
    config,
    travel_time_matrix=matrix,
    grid_mapping=mapping
)
# config['state_encoding']['mode'] に応じて適切なエンコーダを返す
```

---

## 4. RewardDesigner API

### 4.1 初期化

```python
from reinforcement_learning.environment.reward_designer import RewardDesigner

reward_designer = RewardDesigner(config)
```

### 4.2 報酬計算

```python
reward = reward_designer.calculate_step_reward(
    severity='軽症',           # str: 傷病度
    response_time_sec=480.0,  # float: 応答時間（秒）
    L6=0.2,                   # float: 6分カバレッジ損失 (0-1)
    L13=0.1                   # float: 13分カバレッジ損失 (0-1)
)
# Returns: float - 報酬値（-100 ~ 100にクリップ）
```

### 4.3 失敗ペナルティ

```python
penalty = reward_designer.get_failure_penalty('dispatch')
# Args: 'dispatch', 'no_available', 'unhandled'
# Returns: float
```

### 4.4 設定情報取得

```python
info = reward_designer.get_info()
# Returns: Dict
#   - 'hybrid_mode': bool
#   - 'time_weight': float
#   - 'coverage_weight': float
#   - 'critical_params': Dict
#   - 'mild_params': Dict
#   - 'coverage_params': Dict
```

---

## 5. PPOTrainer API

### 5.1 初期化

```python
from reinforcement_learning.training.trainer import PPOTrainer

trainer = PPOTrainer(
    agent=ppo_agent,
    env=ems_env,
    config=config,
    output_dir=Path('experiments/ppo_training/exp1')
)
```

### 5.2 学習実行

```python
trainer.train(start_episode=0)
```

### 5.3 チェックポイント読み込み

```python
trainer.load_checkpoint('path/to/checkpoint_ep1000.pth')
```

### 5.4 ベースライン評価

```python
trainer.run_baseline_evaluation(
    strategy='closest',
    num_episodes=20
)
```

---

## 6. DispatchStrategy API

### 6.1 基底クラス

```python
from dispatch_strategies import DispatchStrategy

class CustomStrategy(DispatchStrategy):
    def __init__(self):
        super().__init__("custom", "rule_based")
    
    def initialize(self, config: Dict):
        """戦略固有の初期化"""
        pass
    
    def select_ambulance(
        self,
        request: EmergencyRequest,
        available_ambulances: List[AmbulanceInfo],
        travel_time_func: callable,
        context: DispatchContext
    ) -> Optional[AmbulanceInfo]:
        """救急車を選択"""
        pass
```

### 6.2 データクラス

```python
@dataclass
class EmergencyRequest:
    id: str
    h3_index: str
    severity: str
    time: float
    priority: DispatchPriority

@dataclass
class AmbulanceInfo:
    id: str
    current_h3: str
    station_h3: str
    status: str
    last_call_time: Optional[float]
    total_calls_today: int

class DispatchContext:
    current_time: float
    hour_of_day: int
    total_ambulances: int
    available_ambulances: int
    all_ambulances: Dict[str, Any]
```

### 6.3 StrategyFactory

```python
from dispatch_strategies import StrategyFactory

# 戦略作成
strategy = StrategyFactory.create_strategy('ppo_agent', {
    'model_path': 'path/to/model.pth',
    'config_path': 'path/to/config.yaml',
    'hybrid_mode': True
})

# 利用可能な戦略一覧
strategies = StrategyFactory.list_available_strategies()
# ['closest', 'closest_distance', 'severity_based', 'ppo_agent', ...]

# カスタム戦略登録
StrategyFactory.register_strategy('custom', CustomStrategy)
```

---

## 7. PPOStrategy API

### 7.1 初期化

```python
from dispatch_strategies import PPOStrategy

ppo_strategy = PPOStrategy()
ppo_strategy.initialize({
    'model_path': 'experiments/ppo_training/exp1/checkpoints/best_model.pth',
    'config_path': 'experiments/ppo_training/exp1/configs/config.yaml',
    'hybrid_mode': True,
    'severe_conditions': ['重症', '重篤', '死亡'],
    'mild_conditions': ['軽症', '中等症']
})
```

### 7.2 救急車選択

```python
selected = ppo_strategy.select_ambulance(
    request,
    available_ambulances,
    travel_time_func,
    context
)
# Returns: AmbulanceInfo or None
```

### 7.3 属性

```python
ppo_strategy.compact_mode  # bool: コンパクトモードフラグ
ppo_strategy.top_k         # int: Top-K数
ppo_strategy.hybrid_mode   # bool: ハイブリッドモードフラグ
ppo_strategy.state_dim     # int: 状態次元
ppo_strategy.action_dim    # int: 行動次元
```

---

## 8. 設定ファイルユーティリティ

### 8.1 継承付き読み込み

```python
from reinforcement_learning.config_utils import load_config_with_inheritance

config = load_config_with_inheritance('experiments/config_hybrid_v8.yaml')
# ベース設定（config.yaml）を継承し、指定ファイルで上書き
```

### 8.2 深いマージ

```python
from reinforcement_learning.config_utils import deep_merge_config

merged = deep_merge_config(base_config, override_config)
```

---

## 9. ValidationSimulator との統合

### 9.1 PPO戦略を使用したシミュレーション

```python
from validation_simulation import run_validation_simulation

run_validation_simulation(
    target_date_str="20240204",
    output_dir="results/ppo_test",
    simulation_duration_hours=24,
    dispatch_strategy="ppo_slot1",
    strategy_config={
        'model_path': 'experiments/ppo_training/exp1/checkpoints/best_model.pth',
        'config_path': 'experiments/ppo_training/exp1/configs/config.yaml',
        'hybrid_mode': True,
        'severe_conditions': ['重症', '重篤', '死亡'],
        'mild_conditions': ['軽症', '中等症']
    },
    enable_visualization=True,
    enable_detailed_reports=True
)
```

---

## 10. 傷病度定数（constants.py）

```python
from constants import (
    SEVERITY_GROUPS,
    SEVERITY_PRIORITY,
    SEVERITY_TIME_LIMITS,
    is_severe_condition,
    is_mild_condition,
    get_severity_time_limit
)

# 傷病度グループ
SEVERITY_GROUPS = {
    'severe_conditions': ['重症', '重篤', '死亡'],
    'mild_conditions': ['軽症', '中等症'],
    'critical_conditions': ['重篤'],
    'moderate_conditions': ['中等症']
}

# 傷病度別時間制限（秒）
SEVERITY_TIME_LIMITS = {
    '重症': 360,   # 6分
    '重篤': 360,
    '死亡': 360,
    '中等症': 780, # 13分
    '軽症': 780
}

# 判定関数
is_severe = is_severe_condition('重症')  # True
time_limit = get_severity_time_limit('軽症')  # 780
```

---

## 更新履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2026-01-23 | 1.0 | 初版作成 |
