# パラメータ設定とwandb連携機能

## 概要

この改良版では、ディスパッチ戦略のパラメータを外部から設定可能になり、wandbとの連携機能が追加されました。

## 主な変更点

### 1. SeverityBasedStrategyクラスの改良

#### 新規追加されたパラメータ
- `time_score_weight`: 応答時間の重み（デフォルト: 0.6）
- `coverage_loss_weight`: カバレッジ損失の重み（デフォルト: 0.4）
- `mild_time_limit_sec`: 軽症の許容時間（秒）（デフォルト: 780秒 = 13分）
- `moderate_time_limit_sec`: 中等症の許容時間（秒）（デフォルト: 780秒 = 13分）

#### 設定方法
```python
config = {
    'time_score_weight': 0.2,            # 応答時間の重みを20%に
    'coverage_loss_weight': 0.8,         # カバレッジ損失の重みを80%に
    'mild_time_limit_sec': 1080,         # 軽症の許容時間を18分(1080秒)に
    'moderate_time_limit_sec': 900       # 中等症の許容時間を15分(900秒)に
}
```

### 2. baseline_comparison.pyの改良

#### wandb連携機能の追加
- 各実験実行が自動的にwandbに記録される
- パラメータ設定もwandbに保存される
- 実験結果の詳細な追跡が可能

#### 設定例
```python
EXPERIMENT_CONFIG = {
    'strategy_configs': {
        'severity_based': {
            'time_score_weight': 0.2,
            'coverage_loss_weight': 0.8,
            'mild_time_limit_sec': 1080,
            'moderate_time_limit_sec': 900
        }
    }
}
```

## 使用方法

### 1. 必要なライブラリのインストール
```bash
pip install -r requirements.txt
```

### 2. wandbの初期設定
```bash
wandb login
```

### 3. 動作確認（推奨）
まず軽量なテストを実行して、wandb連携が正常に動作することを確認します：
```bash
python test_wandb_integration.py
```

### 4. 本格的な実験の実行
```bash
python baseline_comparison.py
```

### 5. トラブルシューティング
もしmatplotlibのエラーが発生する場合は、以下の環境変数を設定してください：
```bash
# Windows
set PYTHONPATH=%PYTHONPATH%;.

# Linux/Mac
export PYTHONPATH=$PYTHONPATH:.
```

## パラメータ調整のガイドライン

### 応答時間 vs カバレッジのバランス調整

#### カバレッジ重視設定（推奨）
```python
{
    'time_score_weight': 0.2,      # 応答時間の重みを低く
    'coverage_loss_weight': 0.8,   # カバレッジ損失の重みを高く
    'mild_time_limit_sec': 1080,   # 軽症は18分まで許容
    'moderate_time_limit_sec': 900 # 中等症は15分まで許容
}
```

#### 応答時間重視設定
```python
{
    'time_score_weight': 0.8,      # 応答時間の重みを高く
    'coverage_loss_weight': 0.2,   # カバレッジ損失の重みを低く
    'mild_time_limit_sec': 600,    # 軽症は10分まで許容
    'moderate_time_limit_sec': 600 # 中等症も10分まで許容
}
```

#### バランス設定
```python
{
    'time_score_weight': 0.5,      # バランス型
    'coverage_loss_weight': 0.5,   # バランス型
    'mild_time_limit_sec': 900,    # 軽症は15分まで許容
    'moderate_time_limit_sec': 780 # 中等症は13分まで許容
}
```

## wandbでの実験追跡

### 記録される情報
1. **パラメータ設定**: 戦略パラメータ、実験設定
2. **実行結果**: 応答時間、達成率、統計情報
3. **実行環境**: 実行時刻、乱数シード、戦略名

### ダッシュボードでの確認
- プロジェクト: `AmbulanceDispatch_Rules`
- グループ化: 戦略名 + 日付でグループ化
- 比較可能: 複数のパラメータ設定を並べて比較

## 注意事項

1. **wandbの利用制限**: 無料アカウントには月間の利用制限があります
2. **パラメータの妥当性**: 極端な値を設定すると予期しない結果になる可能性があります
3. **実験の再現性**: 乱数シードを固定することで実験結果の再現が可能です

## トラブルシューティング

### wandb関連
- ログインエラー: `wandb login` を再実行
- 接続エラー: ネットワーク環境を確認
- 容量制限: 古い実験結果を削除

### パラメータ関連
- 無効な値: 負の値や非数値を設定しない
- 重みの合計: 必要に応じて正規化する
- 時間制限: 現実的な範囲内で設定する
