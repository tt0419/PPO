# Phase2 実装サマリー

## 📋 実装日
2024年11月26日

## 🎯 実装目的
Phase 1で実装したカバレッジ考慮型報酬設計を強化し、アクションマスクとカリキュラム学習を実装することで、より効果的な学習を実現する

## 📝 変更ファイル一覧

### 1. `reinforcement_learning/environment/ems_environment.py`

#### 変更内容
- **アクションマスクの強化**: `get_action_mask()` メソッドを拡張し、時間制約とカバレッジ損失閾値によるフィルタリングを実装
- **統計記録の拡張**: `coverage_loss` と `coverage_component` を `episode_stats` に記録
- **coverage_component計算**: `_calculate_coverage_component_for_stats()` メソッドを追加

#### 主な追加機能

##### アクションマスクの強化（行2151-2225）
```python
def get_action_mask(self) -> np.ndarray:
    """利用可能な行動のマスクを取得（カバレッジ考慮型対応）"""
    # 基本マスク：利用可能な救急車
    
    # coverage_awareモードでアクションマスクが有効な場合、追加フィルタリング
    if (self.reward_designer.mode == 'coverage_aware' and 
        self.pending_call is not None):
        # 軽症系の場合:
        # 1. 時間制約チェック（13分以内）
        # 2. カバレッジ損失閾値チェック（0.8未満）
```

**動作条件**:
- `reward_designer.mode == 'coverage_aware'` の時のみ機能
- `reward.core.action_mask.enabled == true` の時のみ有効
- 重症系の場合は全て許可（時間制約なし）
- 軽症系の場合は時間制約とカバレッジ損失でフィルタリング

##### 統計記録の拡張（行1699-1708）
```python
# coverage_awareモード用の統計記録
if self.reward_designer.mode == 'coverage_aware':
    if 'coverage_loss' not in self.episode_stats:
        self.episode_stats['coverage_loss'] = []
    if 'coverage_component' not in self.episode_stats:
        self.episode_stats['coverage_component'] = []
    
    if 'coverage_loss' in dispatch_result:
        self.episode_stats['coverage_loss'].append(dispatch_result['coverage_loss'])
    if 'coverage_component' in dispatch_result:
        self.episode_stats['coverage_component'].append(dispatch_result['coverage_component'])
```

##### coverage_component計算（行1160-1185）
```python
def _calculate_coverage_component_for_stats(self, severity: str, coverage_loss: float) -> float:
    """統計記録用のcoverage_componentを計算"""
    # 重症系: coverage_weight = 0.0 なので常に0
    # 軽症系: coverage_loss × coverage_scale × coverage_weight
```

#### 変更箇所
- `get_action_mask`: アクションマスクの強化（行2151-2225）
- `_calculate_reward`: `coverage_loss` と `coverage_component` を `dispatch_result` に追加（行1640-1655）
- `_update_statistics`: 統計記録の拡張（行1699-1708）
- `_calculate_coverage_component_for_stats`: 新規メソッド追加（行1160-1185）

---

### 2. `reinforcement_learning/environment/reward_designer.py`

#### 変更内容
- **カリキュラム学習の実装**: エピソード数に応じて報酬重みを動的に変更する機能を追加
- **カリキュラム設定の読み込み**: `curriculum` 設定を読み込む処理を追加

#### 主な追加機能

##### カリキュラム学習の初期化（行126-136）
```python
# ===== カリキュラム学習設定 =====
curriculum_config = config.get('curriculum', {})
self.curriculum_enabled = curriculum_config.get('enabled', False)
self.curriculum_stages = curriculum_config.get('stages', [])
self.current_episode = 0
```

##### カリキュラム学習の更新（行632-680）
```python
def update_curriculum(self, episode: int):
    """カリキュラム学習: エピソード数に応じて報酬重みを更新"""
    # 現在のエピソードに該当するステージを検索
    # severe_params と mild_params を動的に更新
```

**動作条件**:
- `curriculum.enabled == true` の時のみ機能
- `coverage_aware` モードの時のみ有効
- エピソード範囲に応じて `severe_params` と `mild_params` を更新

##### 現在のステージ取得（行682-693）
```python
def get_current_curriculum_stage(self) -> Optional[Dict]:
    """現在のカリキュラムステージを取得"""
    # 現在のエピソードに該当するステージを返す
```

#### 変更箇所
- `__init__`: カリキュラム学習設定の読み込みを追加（行126-136）
- `update_curriculum`: 新規メソッド追加（行632-680）
- `get_current_curriculum_stage`: 新規メソッド追加（行682-693）

---

### 3. `reinforcement_learning/training/trainer.py`

#### 変更内容
- **カリキュラム学習の呼び出し**: 各エピソード開始時にカリキュラム学習を更新
- **WandBログの拡張**: `coverage_loss` と `coverage_component` をWandBに記録

#### 主な追加機能

##### カリキュラム学習の呼び出し（行125-128）
```python
for episode in range(1, self.n_episodes + 1):
    # カリキュラム学習の更新
    if hasattr(self.env, 'reward_designer'):
        self.env.reward_designer.update_curriculum(episode)
    
    # エピソード実行
    ...
```

##### WandBログの拡張（行464-467）
```python
# coverage_awareモード用のメトリクス
if stats and 'coverage_loss' in stats and stats['coverage_loss']:
    log_data['reward/coverage_loss'] = np.mean(stats['coverage_loss'])
if stats and 'coverage_component' in stats and stats['coverage_component']:
    log_data['reward/coverage_component'] = np.mean(stats['coverage_component'])
```

#### 変更箇所
- `train`: カリキュラム学習の呼び出しを追加（行125-128）
- `_log_training_progress`: WandBログの拡張（行464-467）

---

### 4. `reinforcement_learning/experiments/config_coverage_aware_v1.yaml`

#### 変更内容
- **設定ファイルの整理**: `config_continuous.yaml` と同じ構造に整理し、キャプションを追加
- **新規設定の明確化**: 新しく追加した設定項目に「★新規追加」マークを付与

#### 主な変更点
- セクション分けとキャプションを追加
- `severe_params`、`mild_params`、`action_mask` などの新規設定を明確化
- カリキュラム学習設定の整理

---

## 🔧 実装の詳細

### アクションマスクの動作条件

**重要**: アクションマスクの強化機能は、以下の条件を**すべて**満たす場合のみ機能します：

1. `reward_designer.mode == 'coverage_aware'`
2. `reward.core.action_mask.enabled == true`
3. `pending_call` が存在する（事案がある）

**動作**:
- **重症系**: 全ての利用可能な救急車を許可（時間制約なし）
- **軽症系**: 
  - 時間制約: 13分以内の応答時間の救急車のみ許可（`mild_time_limit_mask: true` の場合）
  - カバレッジ損失閾値: 0.8未満の救急車のみ許可（`coverage_loss_mask: true` の場合）
  - フィルタリング後も選択肢がない場合は、元のマスクを返す（最低限の選択肢を確保）

### カリキュラム学習の動作条件

**重要**: カリキュラム学習機能は、以下の条件を**すべて**満たす場合のみ機能します：

1. `curriculum.enabled == true`
2. `reward_designer.mode == 'coverage_aware'`
3. `curriculum.stages` が定義されている

**動作**:
- エピソード数に応じて、設定ファイルの `curriculum.stages` から該当するステージを検索
- 該当するステージの `severe_params` と `mild_params` で `coverage_aware_params` を更新
- 100エピソードごとにログを出力（現在のステージ名と重みを表示）

### WandBログの拡張

**追加されたメトリクス**:
- `reward/coverage_loss`: エピソード平均のカバレッジ損失
- `reward/coverage_component`: エピソード平均のカバレッジコンポーネント

**動作条件**:
- `episode_stats` に `coverage_loss` と `coverage_component` が記録されている場合のみ表示
- `coverage_aware` モードで自動的に記録される

---

## ✅ 動作確認

### 実装済み機能
- ✅ アクションマスクの強化（時間制約とカバレッジ損失閾値）
- ✅ カリキュラム学習の実装（エピソード数に応じた重み変更）
- ✅ WandBログへの `coverage_loss` と `coverage_component` の記録
- ✅ 統計記録への `coverage_loss` と `coverage_component` の追加

### 確認済み項目
- ✅ `coverage_aware` モードでのみ機能することを確認
- ✅ 他の報酬モード（`simple`、`continuous`、`discrete`、`hybrid`）では機能しないことを確認

---

## 📊 期待される効果

### アクションマスクの効果
- 軽症系で不適切な配車（13分超過やカバレッジ損失が大きい）を事前に排除
- 学習の効率化と安定化

### カリキュラム学習の効果
- 段階的な学習により、より安定した収束
- 初期は応答時間のみ学習し、徐々にカバレッジを導入することで、最適解への到達が容易に

### 学習の見通し
- **Stage 1 (ep0-1000)**: カバレッジなし、応答時間のみ学習
- **Stage 2 (ep1000-3000)**: カバレッジを徐々に導入（time_weight: 0.8, coverage_weight: 0.2）
- **Stage 3 (ep3000-5000)**: 最終的なバランス（time_weight: 0.6, coverage_weight: 0.4）

---

## 🔄 次のステップ（Phase3）

1. **高速化**
   - カバレッジ損失計算のキャッシュ化
   - 並列計算の導入

2. **チューニング**
   - ハイパーパラメータの調整
   - ネットワークアーキテクチャの改良

3. **本格的な学習**
   - ep5000程度の長期間学習
   - 各ステージでの性能確認

---

## 📌 注意事項

### アクションマスクについて
- `coverage_aware` モード以外では機能しない（他のモードでは従来通りの動作）
- フィルタリング後も選択肢がない場合は、元のマスクを返す（最低限の選択肢を確保）
- カバレッジ損失計算は計算コストが高いため、アクションマスクで使用する場合は注意が必要

### カリキュラム学習について
- `curriculum.enabled == false` の場合は機能しない
- `coverage_aware` モード以外では機能しない
- エピソード範囲が重複している場合は、最初に見つかったステージが適用される

### WandBログについて
- `coverage_loss` と `coverage_component` は `coverage_aware` モードでのみ記録される
- 他のモードではこれらのメトリクスは表示されない

---

## 🔍 実装の確認方法

### アクションマスクの動作確認
```python
# config_coverage_aware_v1.yaml で以下を設定
reward:
  core:
    mode: coverage_aware
    action_mask:
      enabled: true
      mild_time_limit_mask: true
      coverage_loss_mask: true
      coverage_loss_threshold: 0.8
```

### カリキュラム学習の動作確認
```python
# config_coverage_aware_v1.yaml で以下を設定
curriculum:
  enabled: true
  stages:
    - name: "time_only"
      episodes: [0, 1000]
      mild_params:
        time_weight: 1.0
        coverage_weight: 0.0
    - name: "introduce_coverage"
      episodes: [1000, 3000]
      mild_params:
        time_weight: 0.8
        coverage_weight: 0.2
    - name: "final_balance"
      episodes: [3000, 5000]
      mild_params:
        time_weight: 0.6
        coverage_weight: 0.4
```

### WandBログの確認
- `reward/coverage_loss`: エピソード平均のカバレッジ損失
- `reward/coverage_component`: エピソード平均のカバレッジコンポーネント

---

## 📝 まとめ

Phase 2では、Phase 1で実装したカバレッジ考慮型報酬設計を強化するために、以下の機能を実装しました：

1. **アクションマスクの強化**: 時間制約とカバレッジ損失閾値によるフィルタリング
2. **カリキュラム学習の実装**: エピソード数に応じた報酬重みの動的変更
3. **WandBログの拡張**: `coverage_loss` と `coverage_component` の記録

**重要なポイント**: これらの機能はすべて `coverage_aware` モードでのみ機能し、他の報酬モードでは従来通りの動作を維持します。これにより、既存の設定ファイルとの互換性を保ちながら、新しい機能を段階的に導入できます。








