# 戦略追加・カスタマイズガイド

## 概要
このガイドでは、`baseline_comparison.py`に新しいディスパッチ戦略を追加する方法を説明します。

## 戦略追加の手順

### 1. dispatch_strategies.py での戦略実装

#### 1.1 新しい戦略クラスの作成
```python
class NewStrategy(DispatchStrategy):
    """新しい戦略の説明"""
    
    def __init__(self):
        super().__init__("new_strategy", "rule_based")
        # 戦略固有の初期化
        
    def initialize(self, config: Dict):
        """戦略固有の初期化"""
        # 設定の処理
        
    def select_ambulance(self, request, available_ambulances, travel_time_func, context):
        """救急車選択ロジック"""
        # 実装
        pass
```

#### 1.2 StrategyFactory への登録
```python
class StrategyFactory:
    _strategies = {
        'closest': ClosestAmbulanceStrategy,
        'severity_based': SeverityBasedStrategy,
        'advanced_severity': AdvancedSeverityStrategy,
        'new_strategy': NewStrategy,  # ← 追加
    }
```

#### 1.3 設定プリセットの追加（オプション）
```python
STRATEGY_CONFIGS = {
    "conservative": { ... },
    "aggressive": { ... },
    "extreme": { ... },
    "new_strategy_preset": {  # ← 追加
        'param1': 100,
        'param2': 0.5,
    }
}
```

### 2. baseline_comparison.py での設定変更

#### 2.1 戦略リストへの追加
```python
EXPERIMENT_CONFIG = {
    'strategies': [
        'closest', 
        'severity_based', 
        'advanced_severity',
        'new_strategy'  # ← 追加
    ],
    # ...
}
```

#### 2.2 日本語表示名の追加
```python
'strategy_labels': {
    'closest': '直近隊運用',
    'severity_based': '傷病度考慮運用',
    'advanced_severity': '高度傷病度考慮運用',
    'new_strategy': '新しい戦略名'  # ← 追加
},
```

#### 2.3 色設定の追加
```python
'strategy_colors': {
    'closest': '#3498db',        # 青
    'severity_based': '#e74c3c',  # 赤
    'advanced_severity': '#2ecc71', # 緑
    'new_strategy': '#f39c12'    # オレンジ ← 追加
},
```

#### 2.4 戦略設定の追加
```python
'strategy_configs': {
    'closest': {},
    'severity_based': { ... },
    'advanced_severity': STRATEGY_CONFIGS['aggressive'],
    'new_strategy': {  # ← 追加
        'param1': 100,
        'param2': 0.5,
        # または
        # 'new_strategy': STRATEGY_CONFIGS['new_strategy_preset']
    }
}
```

## 設定変更箇所の詳細

### 【設定変更箇所1】EXPERIMENT_CONFIG
ファイルの上部にある `EXPERIMENT_CONFIG` 辞書を編集します。

```python
EXPERIMENT_CONFIG = {
    # 比較する戦略のリスト（ここで戦略を追加・削除）
    'strategies': ['closest', 'severity_based', 'advanced_severity'],
    
    # 各戦略の日本語表示名
    'strategy_labels': {
        'closest': '直近隊運用',
        'severity_based': '傷病度考慮運用',
        'advanced_severity': '高度傷病度考慮運用'
    },
    
    # 各戦略の色設定
    'strategy_colors': {
        'closest': '#3498db',        # 青
        'severity_based': '#e74c3c',  # 赤
        'advanced_severity': '#2ecc71' # 緑
    },
    
    # 各戦略の設定
    'strategy_configs': {
        'closest': {},
        'severity_based': { ... },
        'advanced_severity': STRATEGY_CONFIGS['aggressive']
    }
}
```

### 【設定変更箇所2】EXPERIMENT_PARAMS
ファイルの下部にある `EXPERIMENT_PARAMS` 辞書を編集します。

```python
EXPERIMENT_PARAMS = {
    'target_date': "20231201",  # 開始日
    'duration_hours': 720,       # 30日間
    'num_runs': 5,              # 各戦略5回実行
    'output_base_dir': 'data/tokyo/experiments'
}
```

## 戦略削除の方法

### 1. 戦略リストから削除
```python
'strategies': ['closest', 'severity_based'],  # advanced_severityを削除
```

### 2. 関連設定も削除
- `strategy_labels` から該当エントリを削除
- `strategy_colors` から該当エントリを削除
- `strategy_configs` から該当エントリを削除

## 色の選択ガイド

### 推奨色パレット
```python
# 基本色
'#3498db'  # 青
'#e74c3c'  # 赤
'#2ecc71'  # 緑
'#f39c12'  # オレンジ
'#9b59b6'  # 紫
'#1abc9c'  # ティール
'#34495e'  # ダークグレー
'#e67e22'  # カロット
```

### 色の組み合わせルール
1. **対比**: 隣接する戦略は異なる色を使用
2. **視認性**: グラフで見分けやすい色を選択
3. **一貫性**: 同じ戦略は常に同じ色を使用

## 戦略設定の詳細

### 設定の種類

#### 1. 空設定（デフォルト）
```python
'closest': {}  # 戦略のデフォルト設定を使用
```

#### 2. カスタム設定
```python
'severity_based': {
    'coverage_radius_km': 5.0,
    'severe_conditions': ['重症', '重篤', '死亡'],
    'mild_conditions': ['軽症', '中等症']
}
```

#### 3. プリセット設定
```python
'advanced_severity': STRATEGY_CONFIGS['aggressive']  # プリセットから選択
```

### 利用可能なプリセット
- `'conservative'`: 保守的設定
- `'aggressive'`: 積極的設定（推奨）
- `'extreme'`: 極端設定（実験用）

## トラブルシューティング

### よくある問題

#### 1. 戦略が見つからないエラー
```
ValueError: Unknown strategy: new_strategy
```
**解決方法**: `dispatch_strategies.py` の `StrategyFactory._strategies` に戦略を登録

#### 2. 設定キーエラー
```
KeyError: 'new_strategy'
```
**解決方法**: `EXPERIMENT_CONFIG` の各辞書に戦略を追加

#### 3. 色が重複している
**解決方法**: `strategy_colors` で異なる色を設定

#### 4. グラフが正しく表示されない
**解決方法**: 
- 戦略数に応じてレイアウトが自動調整されます
- 3つ以下: 2行3列
- 4つ以上: 3行3列

## 高度なカスタマイズ

### 1. 新しい統計指標の追加
`analyze_results` 関数を修正して新しい指標を追加できます。

### 2. 新しい可視化の追加
`visualize_comparison` 関数に新しいグラフを追加できます。

### 3. 統計検定の変更
`create_summary_report` 関数で統計手法を変更できます。

## 例：新しい戦略の追加

### 例1: ワークロード分散戦略の追加
```python
# dispatch_strategies.py
class WorkloadBalancingStrategy(DispatchStrategy):
    def __init__(self):
        super().__init__("workload_balancing", "rule_based")
    
    def select_ambulance(self, request, available_ambulances, travel_time_func, context):
        # ワークロード分散ロジック
        pass

# StrategyFactory に登録
_strategies = {
    'closest': ClosestAmbulanceStrategy,
    'severity_based': SeverityBasedStrategy,
    'advanced_severity': AdvancedSeverityStrategy,
    'workload_balancing': WorkloadBalancingStrategy,  # 追加
}

# baseline_comparison.py
EXPERIMENT_CONFIG = {
    'strategies': ['closest', 'severity_based', 'advanced_severity', 'workload_balancing'],
    'strategy_labels': {
        'closest': '直近隊運用',
        'severity_based': '傷病度考慮運用',
        'advanced_severity': '高度傷病度考慮運用',
        'workload_balancing': 'ワークロード分散運用'  # 追加
    },
    'strategy_colors': {
        'closest': '#3498db',
        'severity_based': '#e74c3c',
        'advanced_severity': '#2ecc71',
        'workload_balancing': '#f39c12'  # 追加
    },
    'strategy_configs': {
        'closest': {},
        'severity_based': { ... },
        'advanced_severity': STRATEGY_CONFIGS['aggressive'],
        'workload_balancing': {  # 追加
            'max_workload_threshold': 10,
            'balance_weight': 0.7
        }
    }
}
```

このガイドに従って、新しい戦略を簡単に追加できます。 