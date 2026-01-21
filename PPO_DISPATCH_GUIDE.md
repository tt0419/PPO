# PPOディスパッチ戦略 実行ガイド

## 概要

改良版PPOディスパッチ戦略を使用して、直近隊運用との性能比較を実施するためのガイドです。

## 前提条件

1. 学習済みPPOモデルが存在すること
   - `models/normal_ppo_20250926_010459.pth` または
   - `models/hybrid_ppo_20250922_034246.pth`

2. 必要なデータファイルが存在すること
   - `data/tokyo/processed/grid_mapping_res9.json`
   - `data/tokyo/calibration2/linear_calibrated_response.npy`

## 実行手順

### 比較実験の実行

`baseline_comparison.py` を使用して、複数回の実験を実施します。

```bash
python baseline_comparison.py
```

**設定の確認:**
- `EXPERIMENT_CONFIG['strategies']` に比較したい戦略が含まれているか
- `EXPERIMENT_CONFIG['strategy_configs']['ppo_agent']` が正しく設定されているか

## 設定パラメータ

### baseline_comparison.py の設定例

```python
EXPERIMENT_CONFIG = {
    'strategies': ['closest', 'ppo_agent'],
    
    'strategy_configs': {
        'closest': {},
        'ppo_agent': {
            'model_path': 'reinforcement_learning/experiments/ppo_training/ppo_XXXXXXXX_XXXXXX/final_model.pth',
            'config_path': 'reinforcement_learning/experiments/ppo_training/ppo_XXXXXXXX_XXXXXX/configs/config.yaml',
            'hybrid_mode': True,  # ハイブリッドモード
            'severe_conditions': ['重症', '重篤', '死亡'],
            'mild_conditions': ['軽症', '中等症']
        }
    }
}

EXPERIMENT_PARAMS = {
    'target_date': "20240801",
    'duration_hours': 720,  # 30日間
    'num_runs': 5,  # 各戦略5回実行
    'wandb_project': 'ems-dispatch-optimization'
}
```

### ハイブリッドモードの使用

重症系は直近隊、軽症系はPPOで選択する場合：

```python
'ppo_agent': {
    'model_path': 'models/hybrid_ppo_20250922_034246.pth',
    'hybrid_mode': True,
    'severe_conditions': ['重症', '重篤', '死亡'],
    'mild_conditions': ['軽症', '中等症']
}
```

## トラブルシューティング

### エラー1: モデルファイルが見つからない

```
FileNotFoundError: Model file not found: models/normal_ppo_20250926_010459.pth
```

**対処法:**
1. `models/` ディレクトリを確認
2. 利用可能なモデルファイルを確認: `ls models/`
3. `baseline_comparison.py` の `model_path` を修正

### エラー2: データファイルが見つからない

```
警告: 移動時間行列が見つかりません
```

**対処法:**
1. データファイルの存在を確認
2. チェックポイント内の `data_paths` 設定を確認
3. 必要に応じてデータファイルを準備

### エラー3: StateEncoderのエラー

```
RuntimeError: StateEncoder initialization failed
```

**対処法:**
1. チェックポイント内の設定を確認
2. `debug_ppo_init.py` で詳細を確認
3. StateEncoderの引数を確認

### エラー4: 次元数の不一致

```
RuntimeError: Expected state_dim=X but got Y
```

**対処法:**
1. チェックポイントから読み込まれる次元数を確認
2. 学習時の設定と一致しているか確認
3. エリア制限の有無を確認

## 改良版PPOの特徴

### 1. 学習時と同じ状態エンコーディング
- StateEncoderを使用して学習時と同じ方法で状態を変換
- チェックポイントから設定を自動読み込み

### 2. ハイブリッドモード対応
- 重症系は直近隊（確実性優先）
- 軽症系はPPO選択（最適化）

### 3. 堅牢性の向上
- エラー時の自動フォールバック
- 詳細なデバッグログ

### 4. 柔軟な設定
- モデルファイルのみ指定すれば動作
- ハイブリッドモードのON/OFF可能

## 結果の確認

### 出力ファイル

```
data/tokyo/experiments/
├── closest_YYYYMMDD_XXXh_runN/
│   └── simulation_report.json
├── ppo_agent_YYYYMMDD_XXXh_runN/
│   └── simulation_report.json
├── strategy_comparison.png
└── comparison_summary.txt
```

### 評価指標

1. **平均応答時間**
   - 全体
   - 重症系
   - 軽症系

2. **閾値達成率**
   - 6分以内（全体）
   - 13分以内（全体）
   - 6分以内（重症系）

3. **統計的有意性**
   - t検定による比較

## 参考情報

- PPO戦略の実装: `dispatch_strategies.py` (777-1047行目)
- テストスクリプト: `test_ppo_dispatch.py`
- デバッグスクリプト: `debug_ppo_init.py`
- 比較実験スクリプト: `baseline_comparison.py`
