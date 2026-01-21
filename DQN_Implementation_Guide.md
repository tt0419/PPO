# 救急車配車最適化プロジェクト：DQN実装ガイド

## ドキュメント概要

**作成日**: 2025年11月14日  
**対象期間**: 2025年11月15日 〜 2025年12月15日  
**目標**: 修士論文のためのDQN実装とPPOとの比較実験完了

---

## 📋 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [現状分析](#2-現状分析)
3. [DQN実装計画](#3-dqn実装計画)
4. [実装スケジュール](#4-実装スケジュール)
5. [次のアクション](#5-次のアクション)
6. [リスク管理](#6-リスク管理)
7. [技術仕様](#7-技術仕様)

---

## 1. プロジェクト概要

### 1.1 研究目的

- **主目的**: 東京都の救急車配車問題に対してDQNを適用し、PPOとの性能比較を行う
- **評価指標**: 
  - 6分達成率（重症系事案）
  - 13分達成率（全事案）
  - 平均応答時間
  - 救治失敗率

### 1.2 現在の状況

✅ **完了済み**
- PPO実装の完成（環境、エージェント、トレーナー）
- ValidationSimulatorの統合
- データパイプライン確立
- 移動時間行列のキャリブレーション完了
- ベースライン比較フレームワーク構築

⚠️ **課題**
- PPO学習の収束が遅い（メモより: Critic損失異常）
- 直近隊運用との性能差が小さい
- 別の強化学習手法との比較が未実施

### 1.3 目標

🎯 **短期目標（12月15日まで）**
- DQN実装の完成
- DQN vs PPO vs ベースライン（直近隊、傷病度考慮）の比較実験
- 結果分析とグラフ作成

🎯 **論文化目標**
- 「確実に修了」を最優先
- 既存環境を最大限活用したマイナーチェンジ実装
- 修論の1章として成立する実験結果の取得

### 1.4 制約条件

| 項目 | 条件 | 理由 |
|------|------|------|
| **H3グリッド粒度** | 3120個（変更不可） | 手法間比較の公平性 |
| **救急車台数** | 192台（変更不可） | 実際の東京都の規模 |
| **消防署・病院** | 位置固定 | 既存データとの整合性 |
| **状態エンコーダ** | 流動的に変更可 | 手法の本質的部分 |
| **報酬関数** | 調整可 | 手法の本質的部分 |

---

## 2. 現状分析

### 2.1 プロジェクト構造

```
project/
├── reinforcement_learning/
│   ├── environment/
│   │   ├── ems_environment.py          # 強化学習環境（PPO用）
│   │   ├── state_encoder.py            # 状態エンコーダ（999次元）
│   │   └── reward_designer.py          # 報酬設計
│   ├── agents/
│   │   ├── ppo_agent.py                # PPOエージェント
│   │   ├── network_architectures.py    # Actor-Critic（流用可）
│   │   └── buffer.py                   # RolloutBuffer（要改修）
│   └── training/
│       └── trainer.py                  # PPOトレーナー（要改修）
├── validation_simulation.py            # テスト環境シミュレーター
├── dispatch_strategies.py              # 配車戦略定義
├── baseline_comparison.py              # ベースライン比較
└── train_ppo.py                        # メイン実行スクリプト
```

### 2.2 既存ファイル詳細分析

#### 2.2.1 buffer.py（RolloutBuffer）

**現状の実装:**
```python
class RolloutBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, device):
        self.states = np.zeros((buffer_size, state_dim))
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size)
        self.next_states = np.zeros((buffer_size, state_dim))
        self.dones = np.zeros(buffer_size, dtype=bool)
        self.log_probs = np.zeros(buffer_size)      # ★PPO専用
        self.values = np.zeros(buffer_size)         # ★PPO専用
        self.action_masks = np.zeros((buffer_size, action_dim))
```

**特徴:**
- エピソード終了ごとにクリア
- 全データを順次使用（GAE計算用）
- log_probs, valuesを保存（PPO専用）

**DQN用ReplayBufferへの変更点:**

| 要素 | PPO | DQN | 変更難易度 |
|------|-----|-----|----------|
| データ構造 | NumPy配列（固定長） | deque（FIFO） | ⭐⭐⭐ |
| log_probs | 必須 | **削除** | ⭐ |
| values | 必須 | **削除** | ⭐ |
| サンプリング | 全データ or 順次 | ランダム | ⭐⭐ |
| クリア | エピソードごと | **なし**（自動上書き） | ⭐ |

**工数見積もり:** 2-3日

---

#### 2.2.2 network_architectures.py

**現状の実装:**
```python
class ActorNetwork(nn.Module):
    # 状態 → 行動確率
    # 入力: [batch, 999] → 出力: [batch, 192] (Softmax)
    
    def __init__(self, state_dim, action_dim, config):
        # ModularStateEncoder使用可能
        self.state_encoder = ModularStateEncoder(max_ambulances=192)
        
        # 隠れ層（設定可能）
        hidden_layers = config['network']['actor']['hidden_layers']  # [128, 64]
        
        # 出力: Softmaxで確率分布化
        self.policy_network = nn.Sequential(...)

class CriticNetwork(nn.Module):
    # 状態 → 状態価値
    # 入力: [batch, 999] → 出力: [batch, 1]
```

**流用可能性分析:**

| 要素 | 流用可否 | 理由 |
|------|---------|------|
| ModularStateEncoder | ✅ 完全流用 | 状態エンコードは共通 |
| 隠れ層構造 | ✅ 完全流用 | MLP構造は同じ |
| 出力層次元 | ✅ そのまま | 192次元は共通 |
| 出力活性化 | ⚠️ 変更必要 | Softmax → なし（Q値は生値） |
| Critic | ❌ 不要 | DQNはQ-Networkのみ |

**DQN用QNetwork設計:**
```python
class QNetwork(nn.Module):
    # 状態 → Q値
    # 入力: [batch, 999] → 出力: [batch, 192] (活性化なし)
    
    def __init__(self, state_dim, action_dim, config):
        # ActorNetworkと同じエンコーダ
        self.state_encoder = ModularStateEncoder(max_ambulances=192)
        
        # ActorNetworkと同じ隠れ層
        hidden_layers = [128, 64]
        
        # 出力: 活性化関数なし（Q値は生値）
        self.q_network = nn.Sequential(...)
```

**工数見積もり:** 1日（ActorNetworkをベースに修正）

---

#### 2.2.3 ppo_agent.py

**現状の主要メソッド:**
```python
class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.actor = ActorNetwork(...)
        self.critic = CriticNetwork(...)
    
    def select_action(self, state, action_mask):
        """確率的行動選択"""
        action_probs = self.actor(state)
        action_probs = action_probs * action_mask
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        return action, log_prob, value
    
    def update(self, buffer):
        """PPOクリッピング更新"""
        # GAE計算
        # PPO損失計算
        # Actor, Critic同時更新
```

**DQN用エージェントへの変更点:**

| メソッド | PPO | DQN | 変更内容 |
|---------|-----|-----|---------|
| select_action | 確率的サンプリング | ε-greedy | 完全書き換え |
| update | PPOクリッピング | ベルマン誤差MSE | 完全書き換え |
| ネットワーク | Actor + Critic | Q + Target-Q | 役割が異なる |

**工数見積もり:** 4日（新規作成）

---

#### 2.2.4 trainer.py（推定）

**現状の実装（推定）:**
```python
class PPOTrainer:
    def train(self):
        for episode in range(n_episodes):
            state = env.reset()
            while not done:
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done = env.step(action)
                buffer.add(state, action, reward, next_state, done, log_prob, value)
                state = next_state
            
            # エピソード終了後に更新
            agent.update(buffer)
            buffer.clear()  # ★PPO特有
```

**DQN用トレーナーへの変更点:**

| 要素 | PPO | DQN | 変更内容 |
|------|-----|-----|---------|
| 更新タイミング | エピソード終了後 | 毎ステップ or N回ごと | ループ内で更新 |
| バッファクリア | 毎回クリア | **クリアしない** | ロジック削除 |
| Target更新 | なし | 定期的にコピー | 新規追加 |

**工数見積もり:** 3日（改修）

---

#### 2.2.5 validation_simulation.py, dispatch_strategies.py

**役割:**
- validation_simulation.py: テスト環境のシミュレーター
- dispatch_strategies.py: 配車戦略の定義（直近隊、傷病度考慮など）

**DQN統合のための追加:**
```python
# dispatch_strategies.py に追加
class RLBasedStrategy(DispatchStrategy):
    """強化学習ベース配車戦略"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def select_ambulance(self, request, available_ambulances):
        state = self._encode_state(request, available_ambulances)
        action = self.agent.select_action(state, training=False)
        return available_ambulances[action]
```

**工数見積もり:** 0.5日

---

### 2.3 流用可能な要素まとめ

| コンポーネント | 流用可否 | 備考 |
|--------------|---------|------|
| EMSEnvironment | ✅ 完全流用 | OpenAI Gym形式、DQNでもそのまま使用可 |
| StateEncoder | ✅ 完全流用 | 999次元状態エンコード |
| RewardDesigner | ✅ 完全流用 | 報酬関数設計 |
| ModularStateEncoder | ✅ 完全流用 | 状態の前処理 |
| ValidationSimulator | ✅ 完全流用 | テスト環境 |
| 移動時間行列 | ✅ 完全流用 | calibration2/*.npy |
| データパイプライン | ✅ 完全流用 | 2023年4月データ |
| 隠れ層構造（MLP） | ✅ 完全流用 | [128, 64]の構造 |
| action_mask機構 | ✅ 完全流用 | 利用可能救急車の制約 |

---

## 3. DQN実装計画

### 3.1 実装アプローチ: マイナーチェンジ

**選択理由:**
1. ✅ 実装コストが最小（13-17日）
2. ✅ 既存環境を最大限活用
3. ✅ 12月中旬完了が現実的
4. ✅ デバッグが容易
5. ✅ 「確実に修了」の方針に合致

**デメリット（認識済み）:**
- 論文手法（2020/2025）との直接比較は困難
- 新規性は限定的
- ただし、修論の成果としては十分

### 3.2 新規作成が必要なファイル

#### 3.2.1 replay_buffer.py

**ファイルパス:** `reinforcement_learning/agents/replay_buffer.py`

**目的:** DQN用の経験回放バッファ

**主要機能:**
- deque構造による自動FIFO
- ランダムサンプリング
- 容量100,000（設定可能）

**実装難易度:** ⭐⭐  
**工数:** 2日

**主要メソッド:**
```python
class ReplayBuffer:
    def __init__(self, capacity=100000, state_dim=999, action_dim=192)
    def push(self, state, action, reward, next_state, done)
    def sample(self, batch_size=32) -> Tuple[Tensor, ...]
    def __len__(self) -> int
```

---

#### 3.2.2 dqn_agent.py

**ファイルパス:** `reinforcement_learning/agents/dqn_agent.py`

**目的:** DQNエージェントの実装

**主要機能:**
- Q-Network + Target Q-Network
- ε-greedy行動選択
- ベルマン誤差による更新
- action_mask対応

**実装難易度:** ⭐⭐⭐  
**工数:** 4日

**主要メソッド:**
```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, config)
    def select_action(self, state, action_mask, training=True) -> int
    def update(self, replay_buffer, batch_size) -> float
    def update_target_network(self)
    def save(self, path)
    def load(self, path)
```

**ハイパーパラメータ:**
- 学習率: 1e-4
- γ（割引率）: 0.99
- ε初期値: 0.9
- ε最小値: 0.01
- ε減衰率: 0.995

---

#### 3.2.3 dqn_trainer.py

**ファイルパス:** `reinforcement_learning/training/dqn_trainer.py`

**目的:** DQN学習の管理

**主要機能:**
- エピソードループ
- 定期的なQ-Network更新
- Target Network更新（10エピソードごと）
- チェックポイント保存
- 統計記録

**実装難易度:** ⭐⭐  
**工数:** 3日

**主要メソッド:**
```python
class DQNTrainer:
    def __init__(self, agent, env, config, output_dir)
    def train(self) -> float
    def save_checkpoint(self, episode, is_best=False)
```

---

#### 3.2.4 train_dqn.py

**ファイルパス:** `train_dqn.py`（プロジェクトルート）

**目的:** DQN学習のメイン実行スクリプト

**主要機能:**
- コマンドライン引数解析
- 設定ファイル読み込み
- 環境・エージェント初期化
- 学習実行
- モデル保存

**実装難易度:** ⭐  
**工数:** 1日

---

#### 3.2.5 config_dqn.yaml

**ファイルパス:** `config_dqn.yaml`（プロジェクトルート）

**目的:** DQN学習の設定ファイル

**主要設定:**
```yaml
experiment:
  name: "dqn_ems_dispatch"
  seed: 42
  device: "cuda"

dqn:
  n_episodes: 1000
  batch_size: 64
  learning_rate: 1.0e-4
  gamma: 0.99
  epsilon_start: 0.9
  epsilon_min: 0.01
  epsilon_decay: 0.995
  update_freq: 4
  target_update_freq: 10
  buffer_size: 100000
  
  network:
    q_network:
      hidden_layers: [128, 64]
      activation: "relu"
      dropout: 0.1
  
  use_modular_encoder: true
  num_ambulances: 192
```

**実装難易度:** ⭐  
**工数:** 0.5日

---

### 3.3 既存ファイルへの追加・変更

#### 3.3.1 network_architectures.py（追加）

**変更内容:** QNetworkクラスの追加

**変更箇所:**
```python
# 既存のActorNetwork, CriticNetworkは保持
# 以下を追加

class QNetwork(nn.Module):
    """DQN用Q-Network"""
    def __init__(self, state_dim, action_dim, config):
        # ActorNetworkと同じエンコーダ・隠れ層を流用
        # 出力層のみ変更（Softmax削除）
    
    def forward(self, state):
        # Q値を出力（活性化関数なし）
```

**実装難易度:** ⭐  
**工数:** 1日

---

#### 3.3.2 dispatch_strategies.py（追加）

**変更内容:** RLBasedStrategyクラスの追加

**変更箇所:**
```python
class RLBasedStrategy(DispatchStrategy):
    """強化学習ベース配車戦略"""
    
    def __init__(self, agent, state_encoder):
        self.agent = agent
        self.state_encoder = state_encoder
    
    def select_ambulance(self, request, available_ambulances, 
                        travel_time_func, context):
        # DQNエージェントで行動選択
        state = self.state_encoder.encode(...)
        action_mask = self._create_mask(available_ambulances)
        action = self.agent.select_action(state, action_mask, training=False)
        return available_ambulances[action]
```

**実装難易度:** ⭐  
**工数:** 0.5日

---

#### 3.3.3 baseline_comparison.py（追加）

**変更内容:** DQN戦略の比較追加

**変更箇所:**
```python
# 比較対象に追加
strategies = {
    'closest': ClosestAmbulanceStrategy(),
    'severity': SeverityBasedStrategy(),
    'ppo': RLBasedStrategy(ppo_agent),      # 既存
    'dqn': RLBasedStrategy(dqn_agent),      # ★追加
}
```

**実装難易度:** ⭐  
**工数:** 0.5日

---

### 3.4 実装の優先順位

```
【Phase 1: コアコンポーネント】優先度: 最高
├─ replay_buffer.py         (2日)
├─ QNetwork追加             (1日)
└─ 単体テスト              (1日)
   合計: 4日

【Phase 2: DQNエージェント】優先度: 最高
├─ dqn_agent.py            (4日)
├─ 単体テスト              (1日)
└─ デバッグ                (1日)
   合計: 6日

【Phase 3: トレーナー統合】優先度: 高
├─ dqn_trainer.py          (3日)
├─ train_dqn.py            (1日)
├─ config_dqn.yaml         (0.5日)
└─ 統合テスト              (1日)
   合計: 5.5日

【Phase 4: 配車戦略統合】優先度: 中
├─ dispatch_strategies追加  (0.5日)
├─ baseline_comparison追加  (0.5日)
└─ テスト                  (1日)
   合計: 2日

【Phase 5: 本格実験】優先度: 高
├─ ハイパーパラメータ調整   (2日)
├─ 比較実験（複数シード）   (2日)
├─ 結果分析                (1日)
└─ グラフ作成              (1日)
   合計: 6日

総工数: 23.5日
```

---

## 4. 実装スケジュール

### 4.1 全体スケジュール（30日間）

```
期間: 2025年11月15日（金）〜 2025年12月15日（日）
目標: 修論実験完了、結果分析完了

Week 1: 11/15(金) - 11/21(木)
├─ Phase 1: コアコンポーネント実装
│  ├─ 11/15-16: replay_buffer.py実装 + 単体テスト
│  ├─ 11/17-18: QNetwork実装 + 単体テスト
│  └─ 11/19-21: 統合テスト、デバッグ
└─ マイルストーン: ReplayBufferとQNetwork完成

Week 2: 11/22(金) - 11/28(木)
├─ Phase 2: DQNエージェント実装
│  ├─ 11/22-24: dqn_agent.py実装（ε-greedy, Target Network）
│  ├─ 11/25-26: 単体テスト
│  └─ 11/27-28: デバッグ、パラメータ調整
└─ マイルストーン: DQNAgent完成

Week 3: 11/29(金) - 12/5(木)
├─ Phase 3: トレーナー統合
│  ├─ 11/29-12/1: dqn_trainer.py実装
│  ├─ 12/2: train_dqn.py + config_dqn.yaml作成
│  ├─ 12/3-4: 統合テスト、初期学習実験（小規模）
│  └─ 12/5: デバッグ、調整
└─ マイルストーン: DQN学習パイプライン完成

Week 4: 12/6(金) - 12/12(木)
├─ Phase 4: 本格実験
│  ├─ 12/6-7: ハイパーパラメータ調整
│  ├─ 12/8-9: DQN vs PPO比較実験（複数シード）
│  ├─ 12/10-11: dispatch_strategies統合、ベースライン比較
│  └─ 12/12: 結果分析、グラフ作成
└─ マイルストーン: 全実験完了

予備日: 12/13(金) - 12/15(日)
├─ バッファ期間
├─ 追加実験（必要に応じて）
└─ 論文執筆開始
```

### 4.2 週次目標

| 週 | 期間 | 目標 | 成果物 |
|----|------|------|--------|
| Week 1 | 11/15-11/21 | コアコンポーネント完成 | replay_buffer.py, QNetwork |
| Week 2 | 11/22-11/28 | DQNエージェント完成 | dqn_agent.py |
| Week 3 | 11/29-12/5 | 学習パイプライン完成 | dqn_trainer.py, train_dqn.py |
| Week 4 | 12/6-12/12 | 実験完了 | 比較実験結果、グラフ |

### 4.3 日次タスク例（Week 1）

```
11/15（金）[Day 1]
□ replay_buffer.py の骨組み作成
□ ReplayBuffer.__init__() 実装
□ ReplayBuffer.push() 実装
□ 動作確認（簡単なテスト）

11/16（土）[Day 2]
□ ReplayBuffer.sample() 実装
□ 単体テスト作成
□ エッジケースのテスト
□ ドキュメント作成

11/17（日）[Day 3]
□ network_architectures.py にQNetwork追加
□ QNetwork.__init__() 実装
□ QNetwork.forward() 実装
□ 動作確認

11/18（月）[Day 4]
□ QNetwork単体テスト
□ ActorNetworkとの比較テスト
□ 重み初期化の確認
□ ドキュメント作成

11/19（火）[Day 5]
□ replay_buffer + QNetwork 統合テスト
□ サンプリング→Q値計算の一連の流れ確認
□ メモリ使用量の確認

11/20（水）[Day 6]
□ デバッグ
□ コードリファクタリング
□ 性能測定（サンプリング速度など）

11/21（木）[Day 7]
□ Week 1のまとめ
□ Week 2の準備
□ コードレビュー
```

---

## 5. 次のアクション

### 5.1 実装ステップ（順番に依頼）

#### **Step 1: ReplayBuffer実装**

**依頼内容:**
> 「replay_buffer.pyの完全な実装コードをお願いします。以下の仕様で作成してください：
> - deque構造
> - 容量100,000（設定可能）
> - push, sample, __len__メソッド
> - 単体テストコード付き」

**成果物:**
- `reinforcement_learning/agents/replay_buffer.py`
- `tests/test_replay_buffer.py`

---

#### **Step 2: QNetwork実装**

**依頼内容:**
> 「network_architectures.pyにQNetworkクラスを追加してください。ActorNetworkをベースに、以下の変更を加えてください：
> - 出力層のSoftmaxを削除（Q値は生値）
> - ModularStateEncoderを流用
> - 隠れ層構造 [128, 64] を流用
> - 単体テストコード付き」

**成果物:**
- `reinforcement_learning/agents/network_architectures.py`（QNetwork追加）
- `tests/test_q_network.py`

---

#### **Step 3: DQNAgent実装**

**依頼内容:**
> 「dqn_agent.pyの完全な実装コードをお願いします。以下の機能を含めてください：
> - ε-greedy行動選択
> - Target Network管理
> - ベルマン誤差による更新
> - action_mask対応
> - save/loadメソッド
> - 単体テストコード付き」

**成果物:**
- `reinforcement_learning/agents/dqn_agent.py`
- `tests/test_dqn_agent.py`

---

#### **Step 4: DQNTrainer実装**

**依頼内容:**
> 「dqn_trainer.pyの完全な実装コードをお願いします。以下の機能を含めてください：
> - エピソードループ
> - 定期的なQ-Network更新（4ステップごと）
> - Target Network更新（10エピソードごと）
> - チェックポイント保存
> - 統計記録（報酬、損失、ε値）」

**成果物:**
- `reinforcement_learning/training/dqn_trainer.py`

---

#### **Step 5: メインスクリプト実装**

**依頼内容:**
> 「train_dqn.pyとconfig_dqn.yamlの実装をお願いします。train_ppo.pyをベースに、DQN用に修正してください。」

**成果物:**
- `train_dqn.py`
- `config_dqn.yaml`

---

#### **Step 6: 配車戦略統合**

**依頼内容:**
> 「dispatch_strategies.pyにRLBasedStrategyを追加し、baseline_comparison.pyにDQN比較を追加してください。」

**成果物:**
- `dispatch_strategies.py`（RLBasedStrategy追加）
- `baseline_comparison.py`（DQN比較追加）

---

### 5.2 テスト方針

#### 5.2.1 単体テスト

各コンポーネントごとに単体テストを作成：

```python
# tests/test_replay_buffer.py
def test_push_and_sample():
    """データの追加とサンプリングをテスト"""
    
def test_capacity_limit():
    """容量制限のテスト（古いデータが削除されるか）"""
    
def test_sample_randomness():
    """ランダムサンプリングのテスト"""
```

#### 5.2.2 統合テスト

```python
# tests/test_dqn_integration.py
def test_full_training_loop():
    """1エピソード分の学習ループをテスト"""
    
def test_target_network_update():
    """Target Networkの更新をテスト"""
    
def test_checkpoint_save_load():
    """チェックポイントの保存・読み込みをテスト"""
```

#### 5.2.3 小規模実験

```bash
# デバッグモード（10エピソード）で動作確認
python train_dqn.py --config config_dqn.yaml --debug
```

---

## 6. リスク管理

### 6.1 技術的リスク

| リスク | 発生確率 | 影響度 | 対策 |
|-------|---------|--------|------|
| **学習が収束しない** | 中 | 高 | 報酬設計調整、学習率調整、小規模実験で検証 |
| **PPOに勝てない** | 中 | 中 | ハイパーパラメータ徹底探索。勝てなくても「比較検証」として論文化可能 |
| **実装バグ** | 中 | 高 | 単体テスト徹底、段階的実装、デバッグログ充実 |
| **メモリ不足** | 低 | 中 | バッファサイズ調整（64GB RAMで十分） |
| **計算時間超過** | 低 | 中 | GPU使用（RTX 4070 Ti SUPER）、バッチサイズ調整 |

### 6.2 スケジュールリスク

| リスク | 発生確率 | 影響度 | 対策 |
|-------|---------|--------|------|
| **実装遅延** | 中 | 高 | Phase 1-2に集中、Phase 4は簡略化可 |
| **デバッグ長期化** | 中 | 高 | 予備日5日確保、小規模実験で早期発見 |
| **実験失敗** | 低 | 中 | 複数シード実行、ベースライン比較で最低限の成果確保 |
| **論文執筆時間不足** | 低 | 高 | 12月13-15日をバッファとして確保 |

### 6.3 リスク対応戦略

#### 最悪のシナリオ
「DQN実装が12月10日までに完了しない」

**対応:**
1. Phase 3まで（DQNTrainer完成）を優先
2. Phase 4（配車戦略統合）は省略
3. 「DQN vs PPO」のみの比較で論文化
4. ベースライン比較は既存のPPO実験結果を流用

#### 学習が収束しないシナリオ
「DQNの性能がランダム行動と変わらない」

**対応:**
1. 報酬関数の調整（シンプル化）
2. 学習率の調整（1e-4 → 1e-3 or 1e-5）
3. εの減衰速度調整
4. それでも改善しない場合は「失敗事例」として論文化（手法の限界考察）

---

## 7. 技術仕様

### 7.1 ハードウェア環境

```
PC: OMEN35L
CPU: Intel Core i7-14700F (最大 5.40GHz / 20コア / 28スレッド / 33MB)
RAM: Kingston FURY 64GB (16GB×4) DDR5-5600MT/s
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
SSD: 2TB PCIe Gen4 NVMe M.2
```

**評価:**
- ✅ GPU: DQN学習に十分（バッチサイズ64で高速学習可能）
- ✅ RAM: 64GB → バッファサイズ100,000でも余裕
- ✅ CPU: 20コア → 並列実験可能

### 7.2 ソフトウェア環境

```python
# requirements.txt（推定）
torch==1.11.0
numpy==1.21.0
pyyaml==6.0
wandb==0.12.0（オプション）
matplotlib==3.5.0
seaborn==0.11.0
```

### 7.3 DQNハイパーパラメータ

| パラメータ | 値 | 根拠 |
|----------|-----|------|
| 学習率 | 1e-4 | Adam最適化、PPOと同等 |
| γ（割引率） | 0.99 | 標準的な値 |
| バッチサイズ | 64 | GPU性能とバランス |
| バッファサイズ | 100,000 | 約50エピソード分 |
| ε初期値 | 0.9 | 初期は探索重視 |
| ε最小値 | 0.01 | 完全にgreedyにしない |
| ε減衰率 | 0.995 | 約1000ステップで最小値到達 |
| 更新頻度 | 4ステップ | 計算コスト削減 |
| Target更新 | 10エピソード | 安定性確保 |

### 7.4 状態空間・行動空間

```
状態空間: 999次元（既存のstatus_encoder.pyを流用）
├─ 救急車特徴: 192台 × 5 = 768次元
│  └─ [位置、利用可能性、出場回数、移動時間]
├─ 事案特徴: 10次元
├─ 時間特徴: 8次元
└─ 空間特徴: 21次元

行動空間: 192次元（離散）
└─ どの救急車を選択するか（事案主体型パラダイム）

action_mask: 192次元（bool）
└─ 利用可能な救急車のみTrue
```

### 7.5 報酬関数

```python
# reward_designer.py を流用
# 既存のPPO用報酬関数をそのまま使用

報酬設計:
- 応答時間ペナルティ: -1.0/分
- 傷病度別重み: 重症系5.0、中等症2.0、軽症1.0
- 閾値ペナルティ: 6分超-5.0、13分超-20.0
- カバレッジ維持報酬: 0.5

（必要に応じて調整）
```

### 7.6 評価指標

```
主要指標:
1. 6分達成率（重症系事案）
2. 13分達成率（全事案）
3. 平均応答時間
4. 救治失敗率

副次指標:
1. 学習曲線（報酬推移）
2. 損失推移
3. ε値推移
4. バッファ使用率
```

---

## 8. 期待される成果

### 8.1 定量的成果

**目標値（DQN vs 直近隊運用）:**
- 救治失敗率: 45.4% → 35%以下
- 6分達成率（重症系）: 38.6% → 45%以上
- 平均応答時間: 7.14分 → 6.5分以下

**比較対象:**
1. 直近隊運用（ベースライン）
2. 傷病度考慮運用（ヒューリスティック）
3. PPO（既存実装）
4. DQN（本実装）

### 8.2 定性的成果

**修論への貢献:**
1. ✅ 複数の強化学習手法（PPO, DQN）の比較検証
2. ✅ 実データに基づく実証実験
3. ✅ 手法間の性能差と特性の分析
4. ✅ 実用化への示唆

**技術的貢献:**
1. ✅ 東京都規模（192台）でのDQN適用実証
2. ✅ action_mask機構の有効性検証
3. ✅ 大規模行動空間でのDQN性能検証

---

## 9. よくある質問（FAQ）

### Q1. PPOとDQNのどちらが性能が良いですか？

**A:** 事前には不明です。一般的にはPPOの方が安定的と言われていますが、問題の性質によってはDQNが勝る可能性もあります。本研究の目的は「比較検証」であり、どちらが勝っても研究成果として価値があります。

### Q2. DQNが収束しない場合はどうしますか？

**A:** 以下の対策を順次試します：
1. 報酬関数の簡略化（ペナルティ緩和）
2. 学習率の調整（1e-4 → 1e-3 or 1e-5）
3. εの減衰速度調整
4. バッチサイズ変更

それでも改善しない場合は「失敗事例」として論文化し、手法の限界を考察します。

### Q3. 論文手法（2020/2025）は実装しないのですか？

**A:** 時間的制約のため、今回は実装しません。ただし、以下の理由で問題ありません：
- 修論の主目的は「確実に修了」
- マイナーチェンジでも十分な研究成果
- 論文手法は将来の拡張として位置づけ可能

### Q4. 実装でつまずいたらどうしますか？

**A:** 以下の手順で対処します：
1. エラーログを確認
2. 単体テストでデバッグ
3. 小規模実験で動作確認
4. このドキュメントの「次のアクション」を参照
5. 必要に応じてClaude AIに質問

---

## 10. 参考資料

### 10.1 論文

1. **Liu et al. (2020)** "Ambulance Dispatch via Deep Reinforcement Learning" (SIGSPATIAL 2020)
   - マルチエージェントDQN
   - グリッド分割アプローチ

2. **Zhang et al. (2025)** "基于在线强化学习算法的救护车智能调控模型" (Journal of System Simulation)
   - DA-DQN（データ拡張）
   - 傷病者集中点モデル

### 10.2 内部ドキュメント

- `20250818_memo.txt`: PPO実装の引き継ぎメモ
- `ディスパッチロジック構築ガイド.txt`: プロジェクト全体の設計書
- `修士論文中間報告書.pdf`: 研究計画と進捗

### 10.3 コードドキュメント

- `README.md`（今後作成予定）
- 各ファイルのdocstring
- テストコードのコメント

---

## 11. 更新履歴

| 日付 | 版 | 変更内容 | 更新者 |
|------|---|----------|--------|
| 2025-11-14 | 1.0 | 初版作成 | Claude AI |

---

## 12. 連絡先・サポート

**プロジェクトリーダー:** 髙橋哲朗（修士2年）  
**指導教員:** 戸田浩之  
**AI アシスタント:** Claude (Anthropic)

**次のアクション依頼方法:**
```
次のチャットで以下のように依頼してください：

「Step 1のreplay_buffer.pyの実装をお願いします。
DQN_Implementation_Guide.mdのStep 1の仕様に従って、
完全な実装コードと単体テストコードを作成してください。」
```

---

**このドキュメントは、次のチャットですぐに実装依頼できるように設計されています。**  
**Step 1から順番に実装を進め、12月15日までに全体を完成させましょう！**

---

END OF DOCUMENT
