# PPOディスパッチ戦略 実験完了報告

## ✅ 実装完了

PPOディスパッチ戦略と直近隊運用の比較実験システムが正常に動作しています。

---

## 📊 実験結果（2024年8月 720時間 × 5回）

### ハイブリッドPPO戦略の性能

実験設定：
- **モデル**: `ppo_20250920_212823`
- **モード**: ハイブリッド（重症系=直近隊、軽症系=PPO）
- **期間**: 2024年8月1日から30日間（720時間）
- **実行回数**: 5回

結果例（Run 5）:
```
重症系平均応答時間: 12.76分
軽症系平均応答時間: 20.64分
中等症平均応答時間: 21.07分
重症系6分以内達成率: 19.30%
```

---

## 🚀 使用方法

### 実験の実行

```bash
python baseline_comparison.py
```

### 設定の変更

`baseline_comparison.py` の以下の箇所を編集：

```python
# 比較する戦略
EXPERIMENT_CONFIG = {
    'strategies': ['closest', 'ppo_agent'],  # ← ここで戦略を選択
    ...
}

# PPO戦略の設定
'ppo_agent': {
    'model_path': 'reinforcement_learning/experiments/ppo_training/ppo_XXXXXXXX_XXXXXX/final_model.pth',
    'config_path': 'reinforcement_learning/experiments/ppo_training/ppo_XXXXXXXX_XXXXXX/configs/config.yaml',
    'hybrid_mode': True,  # ハイブリッドモード ON/OFF
    'severe_conditions': ['重症', '重篤', '死亡'],
    'mild_conditions': ['軽症', '中等症']
}

# 実験パラメータ
EXPERIMENT_PARAMS = {
    'target_date': "20240801",  # 開始日
    'duration_hours': 720,      # シミュレーション期間
    'num_runs': 5,              # 実行回数
    'wandb_project': 'ems-dispatch-optimization'
}
```

---

## 📁 出力ファイル

実験完了後、以下のファイルが生成されます：

### 1. 比較グラフ
`data/tokyo/experiments/strategy_comparison.png`
- 全体平均応答時間
- 重症系平均応答時間
- 軽症系平均応答時間
- 6分以内達成率
- 13分以内達成率
- 重症系6分以内達成率

### 2. 統計レポート
`data/tokyo/experiments/comparison_summary.txt`
- 各戦略の統計情報
- t検定による有意性検定
- 改善率の比較

### 3. 詳細レポート
各実行ごとに：
- `data/tokyo/experiments/closest_YYYYMMDD_XXXh_runN/simulation_report.json`
- `data/tokyo/experiments/ppo_agent_YYYYMMDD_XXXh_runN/simulation_report.json`

### 4. wandb記録
- プロジェクト: `ems-dispatch-optimization`
- 各実行の詳細メトリクス
- リアルタイムでの可視化

---

## 🎯 利用可能な戦略

### 1. closest（直近隊運用）
最も近い救急車を選択

### 2. ppo_agent（PPO運用）
**通常モード**: 全ての事案でPPOが選択
```python
'hybrid_mode': False
```

**ハイブリッドモード**: 重症度で分岐
```python
'hybrid_mode': True,
'severe_conditions': ['重症', '重篤', '死亡'],  # 直近隊
'mild_conditions': ['軽症', '中等症']  # PPO選択
```

### 3. その他の戦略
- `severity_based`: 傷病度考慮運用
- `advanced_severity`: 高度傷病度考慮運用
- `second_ride`: 2番目選択運用
- `mexclp`: MEXCLP運用

---

## 📋 重要な注意事項

### モデルとconfigの両方が必要

❌ **動作しない**:
```python
'model_path': 'models/some_model.pth'
```

✅ **正しい設定**:
```python
'model_path': 'reinforcement_learning/experiments/ppo_training/ppo_XXXXXXXX_XXXXXX/final_model.pth',
'config_path': 'reinforcement_learning/experiments/ppo_training/ppo_XXXXXXXX_XXXXXX/configs/config.yaml'
```

理由：モデルファイル（.pth）のチェックポイントには学習時の設定が完全には保存されていないため、config.yamlが必要です。

---

## 📚 関連ドキュメント

- **クイックスタート**: `QUICK_START_PPO.md`
- **詳細ガイド**: `PPO_DISPATCH_GUIDE.md`
- **戦略カスタマイズ**: `STRATEGY_CUSTOMIZATION_GUIDE.md`
- **パラメータ設定**: `README_parameter_configuration.md`

---

## 🔧 実装の特徴

### PPOStrategyの改良点

1. **学習時と同じ状態エンコーディング**
   - StateEncoderを使用
   - チェックポイントから設定を読み込み

2. **ハイブリッドモード対応**
   - 重症系：直近隊（確実性優先）
   - 軽症系：PPO選択（最適化）

3. **堅牢性の向上**
   - エラー時の自動フォールバック
   - デフォルト設定の提供
   - 詳細なデバッグログ

4. **柔軟な設定**
   - config.yamlからの読み込み
   - ハイブリッドモードのON/OFF
   - 傷病度カテゴリのカスタマイズ

---

## ✨ 今後の展開

### 可能な実験

1. **異なる期間での比較**
   ```python
   'target_date': "20240101",  # 冬季
   'duration_hours': 168       # 1週間
   ```

2. **複数戦略の同時比較**
   ```python
   'strategies': ['closest', 'severity_based', 'ppo_agent', 'mexclp']
   ```

3. **パラメータ感度分析**
   - ハイブリッドモードのON/OFF比較
   - 異なる学習済みモデルの比較

---

## 🎉 まとめ

✅ PPOディスパッチ戦略の実装完了  
✅ 比較実験システムの構築完了  
✅ wandb連携による結果記録  
✅ 詳細な統計分析とレポート生成  

実験は正常に動作しており、いつでも追加の比較実験が実施可能です！
