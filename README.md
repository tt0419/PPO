# 救急隊ディスパッチ最適化プロジェクト

東京都内における救急隊の運用最適化によるレスポンスタイム（RT）改善を目的とした研究プロジェクトです。

## プロジェクト概要

### 研究目的
- 東京都内における救急隊の運用最適化によるレスポンスタイム（RT）改善
- 重症以上の事案を対象とした応答時間短縮
- 強化学習（PPO）をメインとした複数手法の比較検証

### 主な機能
- **シミュレーション**: 救急隊ディスパッチのシミュレーション環境
- **強化学習**: PPO（Proximal Policy Optimization）によるディスパッチ戦略の学習
- **比較実験**: 複数のディスパッチ戦略の性能比較
- **可視化**: 実験結果の可視化とレポート生成

## クイックスタート

### 必要な環境
- Python 3.8以上
- 必要なライブラリは `requirements.txt` を参照

### インストール

```bash
pip install -r requirements.txt
```

### 実行例

#### PPO学習の実行

```bash
python train_ppo.py --config config.yaml
```

#### 比較実験の実行

```bash
python baseline_comparison.py
```

## ディレクトリ構成

```
.
├── reinforcement_learning/    # 強化学習関連
│   ├── agents/                # エージェント実装
│   ├── environment/           # 環境実装
│   └── training/              # トレーニング関連
├── data/                      # データファイル
├── models/                    # 学習済みモデル
├── logs/                      # ログファイル
├── train_ppo.py              # PPO学習メインスクリプト
├── baseline_comparison.py    # 比較実験スクリプト
└── validation_simulation.py  # シミュレーション検証
```

## ドキュメント

詳細なドキュメントは以下のファイルを参照してください：

- **クイックスタートガイド**: `QUICK_START_PPO.md`
- **PPOディスパッチガイド**: `PPO_DISPATCH_GUIDE.md`
- **パラメータ設定**: `README_parameter_configuration.md`
- **プロジェクト引き継ぎ資料**: `project_handover_document.md`

## ライセンス

（必要に応じて追加してください）
