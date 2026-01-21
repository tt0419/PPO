# 救急隊ディスパッチ最適化プロジェクト 引き継ぎ資料

## プロジェクト概要

### 研究目的
- 東京都内における救急隊の運用最適化によるレスポンスタイム（RT）改善
- 重症以上の事案を対象とした応答時間短縮
- 強化学習（PPO）をメインとした複数手法の比較検証

### 現在の進捗状況
- 基本シミュレーションモデル完成（validation_simulation.py）
- 実測値に近い統計値・分布の再現を達成
- 病院選択モデル、移動時間行列、サービス時間生成器が実装済み
- 現行ディスパッチ（最寄り救急車）の動作確認完了

### 次のフェーズ
- ディスパッチ戦略の多様化（ルールベース、強化学習、最適化手法）
- 複数戦略の比較検証システム構築
- 論文用の統計的分析・可視化機能実装

## システム設計

### ファイル構成
```
project/
├── validation_simulation.py          # 既存：メインシミュレータ（要修正）
├── dispatch_strategies.py           # 新規：戦略インターフェース + ルールベース
├── reinforcement_learning/          # 新規：強化学習関連
│   ├── ppo_agent.py                # PPOエージェント
│   ├── dqn_agent.py                # DQNエージェント（対抗馬）
│   ├── a3c_agent.py                # A3Cエージェント（対抗馬）
│   ├── training_environment.py     # 強化学習用環境
│   └── models/                     # 学習済みモデル格納
├── optimization/                   # 新規：最適化手法
│   ├── linear_programming.py      # 線形計画法
│   ├── genetic_algorithm.py       # 遺伝的アルゴリズム
│   └── simulated_annealing.py     # 焼きなまし法
├── evaluation/                    # 新規：比較評価・論文用分析
│   ├── comparative_analysis.py    # 戦略間統計的比較
│   └── visualization.py           # 高品質可視化・論文用図表
└── experiments/                   # 新規：実験スクリプト
    ├── baseline_comparison.py     # ベースライン比較実験
    ├── sensitivity_analysis.py    # 感度分析
    └── large_scale_experiment.py  # 大規模比較実験
```

### 機能分担

#### dispatch_strategies.py（最重要）
**役割：**
- 全戦略の共通インターフェース定義
- ルールベース戦略群の実装
- 戦略の動的生成・管理
- validation_simulatorとの統合ハブ

**実装する戦略：**
- ClosestAmbulanceStrategy：現行（最寄り）
- SeverityBasedStrategy：傷病度考慮
- WorkloadBalancingStrategy：ワークロード分散
- CoverageBasedStrategy：カバレッジ重視
- HybridRuleStrategy：複合ルール

#### reinforcement_learning/
**役割：**
- 強化学習エージェントの実装
- 学習環境の定義
- 学習済みモデルの管理

**実装する手法：**
- PPOAgent：メイン手法
- DQNAgent：対抗馬1
- A3CAgent：対抗馬2

#### optimization/
**役割：**
- 数理最適化手法の実装
- メタヒューリスティクス

**実装する手法：**
- LinearProgrammingDispatcher：線形計画法
- GeneticAlgorithmDispatcher：遺伝的アルゴリズム
- SimulatedAnnealingDispatcher：焼きなまし法

#### evaluation/
**役割：**
- 複数戦略の統計的比較分析
- 論文用高品質可視化
- 政策提案用レポート生成

**現在のvalidation出力との違い：**
- validation：単一シミュレーション結果の基本統計
- evaluation：複数戦略間の比較分析、統計的有意性検定、効果量計算

## 実装の優先順位とアプローチ

### Phase 1: 基盤構築（最優先）
1. **dispatch_strategies.py の実装**
   - DispatchStrategy抽象クラス
   - StrategyFactory（戦略動的生成）
   - 基本ルールベース戦略（closest, severity_based）

2. **validation_simulation.py の修正**
   - find_closest_available_ambulance()の置き換え
   - DispatchManagerとの統合
   - 後方互換性の維持

### Phase 2: ルールベース戦略開発
1. **SeverityBasedStrategy の詳細実装**
   - 重症：最寄り + 距離制限
   - 軽症：カバレッジ考慮 + 稼働率判定

2. **WorkloadBalancingStrategy の実装**
   - 応答時間とワークロードの重み付け合計最小化

3. **初期比較実験**
   - closest vs severity_based の性能比較

### Phase 3: 強化学習実装
1. **PPOAgent の実装**
   - 状態設計（時刻、場所、傷病度、救急車状況）
   - 行動設計（救急車選択確率）
   - 報酬設計（応答時間、重症優先度）

2. **TrainingEnvironment の実装**
   - 学習用シミュレーション環境
   - 経験データ蓄積機能

3. **学習と評価**
   - PPO vs ルールベース比較

### Phase 4: 対抗手法と包括比較
1. **DQN、最適化手法の実装**
2. **evaluation/ モジュール開発**
3. **大規模比較実験実行**

## 重要な技術仕様

### DispatchStrategy インターフェース
```python
class DispatchStrategy(ABC):
    def __init__(self, name: str, strategy_type: str):
        self.name = name
        self.strategy_type = strategy_type  # 'rule_based', 'reinforcement_learning', 'optimization'
        self.metrics = {}
        self.config = {}
    
    @abstractmethod
    def select_ambulance(self, request, available_ambulances, travel_time_func, context):
        """救急車を選択する"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict):
        """戦略固有の初期化"""
        pass
    
    def requires_training(self) -> bool:
        """学習が必要かどうか"""
        return self.strategy_type in ['reinforcement_learning', 'optimization']
```

### EmergencyRequest データクラス
```python
@dataclass
class EmergencyRequest:
    id: str
    h3_index: str
    severity: str
    time: float
    priority: DispatchPriority  # CRITICAL, HIGH, MEDIUM, LOW
    estimated_survival_rate: float = 1.0
    
    def get_urgency_score(self) -> float:
        """緊急度スコアを計算"""
```

### AmbulanceInfo データクラス
```python
@dataclass
class AmbulanceInfo:
    id: str
    current_h3: str
    status: str
    last_call_time: Optional[float] = None
    total_calls_today: int = 0
    current_workload: float = 0.0
```

### ValidationSimulator の主要修正点
```python
class ValidationSimulator:
    def __init__(self, ..., dispatch_strategy: str = 'closest', strategy_config: Dict = None):
        # 戦略の動的生成
        self.dispatch_strategy = StrategyFactory.create_strategy(dispatch_strategy, strategy_config or {})
    
    def find_closest_available_ambulance(self, call_h3: str) -> Optional[Ambulance]:
        """新しいディスパッチロジック"""
        # 1. available_ambulancesをAmbulanceInfoに変換
        # 2. EmergencyRequestを構築
        # 3. contextを構築（時刻、稼働率など）
        # 4. self.dispatch_strategy.select_ambulance()を呼び出し
        # 5. 結果をAmbulanceオブジェクトに変換して返却
```

## 実験・評価の計画

### 比較対象戦略
1. **closest**：現行（ベースライン）
2. **severity_based**：傷病度考慮（ルールベース改良）
3. **ppo**：PPO強化学習（メイン提案手法）
4. **dqn**：DQN強化学習（対抗馬）
5. **genetic_algorithm**：遺伝的アルゴリズム（最適化手法）

### 評価指標
**主要指標：**
- 重症事案6分以内到着率
- 重症事案平均応答時間
- 全事案13分以内到着率

**副次指標：**
- 救急車稼働率
- 地域別応答時間格差
- 時間帯別性能変化

### 実験設計
**統計的検定：**
- 各戦略10回実行（異なる乱数シード）
- t検定、Wilcoxon検定による有意性検証
- 効果量（Cohen's d）計算

**シナリオ別分析：**
- 高需要期 vs 通常期
- 時間帯別（日中/夜間/深夜）
- 地域別（都心部/住宅地/郊外）

## データファイルと依存関係

### 既存の重要ファイル
- `data/tokyo/import/hanso_special_wards.csv`：救急搬送実績データ
- `data/tokyo/import/amb_place_master.csv`：救急署マスタ
- `data/tokyo/import/hospital_master.csv`：病院マスタ
- `data/tokyo/calibration2/`：キャリブレーション済み移動時間行列
- `data/tokyo/processed/hospital_selection_model_revised.pkl`：病院選択確率モデル
- `data/tokyo/service_time_analysis/`：サービス時間パラメータ

### 新規作成予定ファイル
- `models/ppo_trained_model.pkl`：学習済みPPOモデル
- `experiments/comparative_results/`：比較実験結果
- `evaluation/statistical_analysis/`：統計分析結果
- `evaluation/figures/`：論文用図表

## 開発時の注意点

### 既存コードとの互換性
- validation_simulation.pyの既存機能を破壊しない
- 現在の出力形式を維持（evaluation/で拡張）
- 既存のパラメータ設定を引き継ぐ

### パフォーマンス考慮
- 強化学習の状態設計は軽量に（リアルタイム選択のため）
- 大規模実験時のメモリ使用量に注意
- 並列実行対応（複数戦略の同時実験）

### 実験の再現性
- 乱数シードの管理
- 設定パラメータの記録
- 実験環境の文書化

## 成果物（修士論文用）

### 技術的貢献
1. 複雑都市環境での救急ディスパッチ最適化
2. 傷病度考慮型ルールベース戦略の提案
3. PPOベース強化学習手法の設計・実装

### 実証的貢献
1. 東京都実データでの包括的性能比較
2. 統計的有意性を伴う効果検証
3. 地域・時間帯別の詳細分析

### 政策的貢献
1. 東京消防庁への具体的改善提案
2. 実装可能性を考慮した運用モデル
3. 費用対効果の定量的評価

## 次回開発時のアクションアイテム

### 最優先タスク
1. dispatch_strategies.pyの基本実装
2. validation_simulation.pyのintegration部分修正
3. closest vs severity_basedの初期比較実験

### 開発順序
1. DispatchStrategy抽象クラス → ClosestAmbulanceStrategy → SeverityBasedStrategy
2. StrategyFactory実装
3. ValidationSimulatorの統合テスト
4. 基本比較実験実行

### 検証項目
- 既存のvalidation結果と新システムの結果が一致すること
- 戦略切り替えが正常に動作すること
- メトリクス収集が適切に行われること

---

**このドキュメントを参照して、段階的にシステム拡張を進めてください。各Phase完了時に動作確認を行い、問題があれば前のPhaseに戻って修正することを推奨します。**