# config.yaml 設定追加ガイド

## 概要

コンパクト状態空間モードを有効化するための設定を追加する。

---

## 追加する設定

以下のセクションを config.yaml に追加:

```yaml
# ============================================================
# 状態空間エンコーディング設定
# ============================================================
state_encoding:
  # モード選択
  # - 'full': 従来の999次元（192台×5特徴量 + 事案10 + 時間8 + 空間21）
  # - 'compact': コンパクト37次元（傷病度2 + Top-K×3 + グローバル5）
  mode: 'compact'
  
  # Top-K設定（compactモード時のみ有効）
  top_k: 10  # 考慮する上位救急車数（推奨: 10）
  
  # 正規化パラメータ
  normalization:
    max_travel_time_minutes: 30   # 移動時間の上限（これを1.0とする）
    max_station_distance_km: 10   # 署距離の上限（これを1.0とする）
```

---

## 設定の配置場所

既存の config.yaml の構造に応じて、適切な場所に配置:

### パターン1: トップレベルに追加

```yaml
# config.yaml

experiment:
  name: "ppo_compact_state"
  device: "cuda"
  seed: 42

# ★★★ 追加 ★★★
state_encoding:
  mode: 'compact'
  top_k: 10
  normalization:
    max_travel_time_minutes: 30
    max_station_distance_km: 10

ppo:
  n_episodes: 8000
  ...

data:
  ...
```

### パターン2: data セクション内に追加

```yaml
data:
  train_periods:
    - start_date: '20230115'
      end_date: '20230121'
  
  # ★★★ 追加 ★★★
  state_encoding:
    mode: 'compact'
    top_k: 10
```

---

## モード切り替えの例

### コンパクトモード（推奨）

```yaml
state_encoding:
  mode: 'compact'
  top_k: 10
```

結果:
- 状態次元: 37
- 行動次元: 10
- 学習効率: 大幅改善

### 従来モード

```yaml
state_encoding:
  mode: 'full'
```

または、`state_encoding` セクション自体を省略（デフォルトは 'full'）。

結果:
- 状態次元: 999
- 行動次元: 192
- 学習効率: 従来通り

---

## top_k の選択ガイドライン

| top_k | 状態次元 | 行動次元 | 推奨用途 |
|-------|----------|----------|----------|
| 5 | 22 | 5 | 超高速学習、限定的な選択肢 |
| 10 | 37 | 10 | **推奨**。バランスが良い |
| 15 | 52 | 15 | やや多めの選択肢 |
| 20 | 67 | 20 | 傷病度考慮運用に近い候補数 |

**推奨: top_k = 10**

理由:
1. 傷病度考慮運用の候補数（時間制限内）に近い
2. 学習効率と選択肢のバランスが良い
3. 10クラス分類は深層学習で一般的

---

## 完全な設定例

```yaml
# config.yaml - コンパクトモード用

experiment:
  name: "ppo_compact_v1"
  device: "cuda"
  seed: 42

# 状態空間エンコーディング
state_encoding:
  mode: 'compact'
  top_k: 10
  normalization:
    max_travel_time_minutes: 30
    max_station_distance_km: 10

# PPO設定
ppo:
  n_episodes: 8000
  batch_size: 64
  learning_rate:
    actor: 0.0003
    critic: 0.001
  clip_epsilon: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  n_epochs: 10
  entropy_coef: 0.02  # 少し高めに設定（探索促進）

# データ設定
data:
  train_periods:
    - start_date: '20230115'
      end_date: '20230121'
  validation_periods:
    - start_date: '20230122'
      end_date: '20230128'
  episode_duration_hours: 24

# 傷病度設定
severity:
  categories:
    critical:
      conditions: ['重症', '重篤', '死亡']
      reward_weight: 1.0
    moderate:
      conditions: ['中等症']
      reward_weight: 1.0
    mild:
      conditions: ['軽症']
      reward_weight: 1.0
  thresholds:
    golden_time: 360   # 6分 = 360秒
    standard_time: 780 # 13分 = 780秒

# 報酬設定
reward:
  core:
    mode: 'continuous'
  system:
    dispatch_failure: -1.0
    no_available_ambulance: 0.0

# カバレッジ設定
coverage_params:
  time_threshold_seconds: 600  # 10分

# 学習設定
training:
  checkpoint_interval: 500
  early_stopping:
    enabled: true
    patience: 20
    min_delta: 0.01
  logging:
    tensorboard: false
    wandb: false

# 評価設定
evaluation:
  interval: 100
  n_eval_episodes: 5
```

---

## 動作確認

設定が正しく読み込まれているか確認するログ出力:

```
★ コンパクトモード有効: Top-10
  状態次元: 37
  行動次元: 10
```

従来モードの場合:

```
状態空間次元: 999
行動空間次元: 192
```
