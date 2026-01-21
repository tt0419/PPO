# Phase1 実装サマリー

## 📋 実装日
2024年11月25日

## 🎯 実装目的
カバレッジ損失を報酬に組み込むことで、傷病度考慮運用と同等の性能を達成するための基盤実装

## 📝 変更ファイル一覧

### 1. `reinforcement_learning/environment/reward_designer.py`

#### 変更内容
- **新規モード追加**: `coverage_aware` モードを追加
- **パラメータ読み込み**: `reward.core.severe_params` と `reward.core.mild_params` を読み込む処理を追加
- **報酬計算関数**: `_calculate_coverage_aware_reward()` メソッドを実装

#### 主な追加機能
```python
def _calculate_coverage_aware_reward(self, severity, response_time_minutes, coverage_loss):
    """カバレッジ考慮型モードの報酬計算"""
    # 重症系: 応答時間のみ（time_weight × 3.0）
    # 軽症系: 応答時間 × 0.6 + カバレッジ損失 × 0.4
```

#### 変更箇所
- `__init__`: `coverage_aware_params` の読み込み処理を追加（行48-81付近）
- `calculate_step_reward`: `coverage_aware` モードの分岐を追加（行336-337）
- `_calculate_coverage_aware_reward`: 新規メソッド追加（行508-553）

---

### 2. `reinforcement_learning/environment/ems_environment.py`

#### 変更内容
- **カバレッジ損失計算**: `_calculate_coverage_loss()` メソッドを実装
- **サンプルポイント取得**: `_get_coverage_sample_points_for_loss()` メソッドを追加
- **最小応答時間計算**: `_get_min_response_time_for_coverage()` メソッドを追加
- **簡易カバレッジ損失**: `_simple_coverage_loss()` メソッドを追加
- **配車処理の拡張**: `_dispatch_ambulance()` に利用可能救急車のスナップショットを保存
- **報酬計算の拡張**: `_calculate_reward()` でカバレッジ損失を計算してRewardDesignerに渡す

#### 主な追加機能
```python
def _calculate_coverage_loss(self, selected_ambulance_id, available_ambulances_before, request_h3):
    """選択した救急車によるカバレッジ損失を計算（0-1の範囲）"""
    # 6分カバレッジと13分カバレッジの変化を計算
    # 重み付け合成（50% + 50%）
```

#### 変更箇所
- `_calculate_coverage_loss`: 新規メソッド追加（行1054-1115）
- `_get_coverage_sample_points_for_loss`: 新規メソッド追加（行1117-1129）
- `_get_min_response_time_for_coverage`: 新規メソッド追加（行1131-1144）
- `_simple_coverage_loss`: 新規メソッド追加（行1146-1157）
- `_dispatch_ambulance`: 利用可能救急車のスナップショット保存を追加（行1186-1220付近）
- `_calculate_reward`: カバレッジ損失の計算とRewardDesignerへの受け渡しを追加（行1632-1654）

---

## 🔧 実装の詳細

### カバレッジ損失の計算ロジック

1. **サンプルポイントの取得**
   - 選択された救急車のステーション位置（H3インデックス）を中心に
   - H3リング2以内のグリッドから最大20ポイントをサンプリング
   - `grid_mapping` に存在するグリッドのみを使用

2. **カバレッジ率の計算**
   - 各サンプルポイントについて、出動前と出動後の最小応答時間を計算
   - 6分以内（360秒）と13分以内（780秒）のカバレッジ率を算出
   - カバレッジ損失 = (出動前カバレッジ - 出動後カバレッジ) / サンプル数

3. **重み付け合成**
   - 6分カバレッジ損失 × 0.5 + 13分カバレッジ損失 × 0.5
   - 0-1の範囲にクリップ

### 報酬計算の分岐

- **重症系**（重症・重篤・死亡）:
  - 応答時間のみを重視（`time_weight: 3.0`）
  - カバレッジは考慮しない（`coverage_weight: 0.0`）

- **軽症系**（軽症・中等症）:
  - 応答時間 × 0.6 + カバレッジ損失 × 0.4
  - 傷病度別の重み（軽症: 0.5、中等症: 1.5）

---

## ✅ 動作確認

### 実装済み機能
- ✅ カバレッジ損失の計算が可能
- ✅ 報酬計算にカバレッジ損失が反映される
- ✅ 重症系と軽症系で報酬計算が分岐する
- ✅ 設定ファイルからパラメータを読み込める

### 確認が必要な項目
- ⚠️ wandbログへの記録（`reward/coverage_component`, `coverage/mean_loss` など）
- ⚠️ カバレッジ損失計算のパフォーマンス（サンプルサイズ20で許容範囲か）

---

## 📊 期待される効果

### 定量的な目標
- 重症系平均RT: 10.5分以下（直近隊: 10.87分を上回る）
- 重症系6分達成率: 22%以上（直近隊: 20.8%を上回る）

### 学習の見通し
- **Stage 1 (ep0-1000)**: カバレッジなし、応答時間のみ学習
- **Stage 2 (ep1000-3000)**: カバレッジを徐々に導入
- **Stage 3 (ep3000-5000)**: 最終的なバランス（time 60% + coverage 40%）

---

## 🔄 次のステップ（Phase2）

1. **アクションマスクの強化**
   - 軽症系で13分以内かつカバレッジ損失が閾値以下の候補のみ許可

2. **カリキュラム学習の実装**
   - エピソード数に応じて重みを段階的に変更

3. **wandbログの拡張**
   - `reward/time_component`, `reward/coverage_component` を記録
   - `coverage/mean_loss`, `coverage/6min_rate`, `coverage/13min_rate` を記録

---

## 📌 注意事項

- カバレッジ損失計算は計算コストが高い（サンプルサイズ20ポイント）
- 必要に応じてサンプルサイズを調整可能（`mild_params.sample_points`）
- カバレッジ損失が計算できない場合は簡易計算にフォールバック

