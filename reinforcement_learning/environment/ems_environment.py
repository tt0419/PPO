"""
ems_environment.py
救急隊ディスパッチのための強化学習環境
"""

import numpy as np
import torch
import yaml
import json
import h3
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import pickle
import random

# 現在のプロジェクトディレクトリ（05_Ambulance_RL）を取得
# ファイル構造: 05_Ambulance_RL/reinforcement_learning/environment/ems_environment.py
CURRENT_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(CURRENT_PROJECT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_PROJECT_DIR))

# 後方互換性のため fix_dir も同じディレクトリを参照
fix_dir = CURRENT_PROJECT_DIR

# 親ディレクトリ（必要な場合のみ）
PROJECT_ROOT = CURRENT_PROJECT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 必要なモジュールをインポート
from data_cache import get_emergency_data_cache
from constants import SEVERITY_GROUPS, is_severe_condition, get_severity_time_limit

# ★★★ v13: 事前計算カバレッジ計算モジュール ★★★
from .station_coverage_calculator import StationCoverageCalculator

# ServiceTimeGeneratorEnhancedのインポート
# 現在のプロジェクトディレクトリ（05_Ambulance_RL）を取得
CURRENT_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent  # 05_Ambulance_RL ディレクトリ
service_time_analysis_path = CURRENT_PROJECT_DIR / "data" / "tokyo" / "service_time_analysis"
sys.path.append(str(service_time_analysis_path))
try:
    from service_time_generator_enhanced import ServiceTimeGeneratorEnhanced
    USE_ENHANCED_GENERATOR = True
except ImportError:
    print("警告: ServiceTimeGeneratorEnhancedが見つかりません。従来版を使用します。")
    print(f"検索パス: {service_time_analysis_path}")
    USE_ENHANCED_GENERATOR = False

# validation_simulation のインポート（fix_dirから）
from validation_simulation import (
    ValidationSimulator,
    EventType,
    AmbulanceStatus,
    EmergencyCall,
    Event,
    ServiceTimeGenerator
)

# 設定ユーティリティのインポート
try:
    from ..config_utils import load_config_with_inheritance
except ImportError:
    # スタンドアロン実行時のフォールバック
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_utils import load_config_with_inheritance

@dataclass
class StepResult:
    """ステップ実行結果"""
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]

class EMSEnvironment:
    """
    PPO学習用のEMS環境
    OpenAI Gym形式のインターフェースを提供
    """
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "train"):
        """
        Args:
            config_path: 設定ファイルのパス
            mode: "train" or "eval"
        """
        # 設定読み込み（継承機能付き）
        self.config = load_config_with_inheritance(config_path)
        
        self.mode = mode
        self.current_period_idx = 0
        
        # ログ制御フラグ
        self._first_period_logged = False
        self._episode_count = 0
        
        print("=" * 60)
        print(f"EMS環境初期化 (モード: {mode})")
        print(f"設定ファイル: {config_path}")
        print("=" * 60)
        
        # データキャッシュの初期化
        print("データキャッシュを初期化中...")
        self.data_cache = get_emergency_data_cache()
        
        # 初回データ読み込み（起動時に一度だけ）
        print("初期データ読み込み中...")
        self.data_cache.load_data()
        print("データキャッシュ準備完了")
        

        
        # 傷病度設定の初期化
        self._setup_severity_mapping()
        
        # データパスの設定（現在のプロジェクトディレクトリからの絶対パスを使用）
        self.base_dir = CURRENT_PROJECT_DIR / "data" / "tokyo"
        print(f"プロジェクトディレクトリ: {CURRENT_PROJECT_DIR}")
        print(f"データベースディレクトリ: {self.base_dir}")
        if not self.base_dir.exists():
            print(f"警告: データディレクトリが存在しません: {self.base_dir}")
        self._load_base_data()
        
        # 移動時間行列の読み込み（ValidationSimulatorと同じ方法）
        self.travel_time_matrices = {}
        self.travel_distance_matrices = {}
        
        calibration_dir = self.base_dir / "calibration2"
        travel_time_stats_path = calibration_dir / 'travel_time_statistics_all_phases.json'
        
        if travel_time_stats_path.exists():
            with open(travel_time_stats_path, 'r', encoding='utf-8') as f:
                phase_stats_data = json.load(f)
            
            # ValidationSimulatorと同じロジックで行列を読み込み
            for phase in ['response', 'transport', 'return']:
                matrix_filename = None
                
                if phase in phase_stats_data and 'calibrated' in phase_stats_data[phase]:
                    model_type = phase_stats_data[phase]['calibrated'].get('model_type')
                    
                    if model_type == "uncalibrated":
                        matrix_filename = f"uncalibrated_travel_time_{phase}.npy"
                    elif model_type in ['linear', 'log']:
                        matrix_filename = f"{model_type}_calibrated_{phase}.npy"
                    
                    if matrix_filename:
                        matrix_path = calibration_dir / matrix_filename
                        if matrix_path.exists():
                            self.travel_time_matrices[phase] = np.load(matrix_path)
                            print(f"  移動時間行列読み込み: {phase} ({model_type})")
        
        # 距離行列も同様に読み込み
        distance_matrix_path = self.base_dir / "processed/travel_distance_matrix_res9.npy"
        if distance_matrix_path.exists():
            travel_distance_matrix = np.load(distance_matrix_path)
            # ValidationSimulatorと同じ形式に変換
            self.travel_distance_matrices = {
                'dispatch_to_scene': travel_distance_matrix,
                'scene_to_hospital': travel_distance_matrix,
                'hospital_to_station': travel_distance_matrix
            }
        
        # シミュレータの初期化は reset() で行う
        self.simulator = None
        self.current_episode_calls = []
        self.pending_call = None
        self.episode_step = 0
        self.max_steps_per_episode = None
        
        # デバッグ用のverbose_logging属性を初期化
        self.verbose_logging = False
        
        # 教師一致情報の初期化
        self.current_matched_teacher = False
        
        # ========== コンパクトモードの設定 ==========
        state_encoding_config = self.config.get('state_encoding', {})
        self.compact_mode = state_encoding_config.get('mode', 'full') == 'compact'
        self.top_k = state_encoding_config.get('top_k', 10)
        
        # 移動時間行列の取得
        response_matrix = self.travel_time_matrices.get('response', None)
        if response_matrix is None:
            print("警告: responseフェーズの移動時間行列が見つかりません。")
        
        # ========== 状態・行動空間の次元設定 ==========
        if self.compact_mode:
            # コンパクトモード: action_dim = top_k, state_dim = 46
            self.action_dim = self.top_k
            
            # ★★★ v13: 状態エンコーディングのバージョンを確認 ★★★
            encoding_version = state_encoding_config.get('version', 'v1')
            
            if encoding_version == 'v2':
                # v2: 決定論的カバレッジ計算（事前計算方式）
                coverage_config = state_encoding_config.get('coverage_calculation', {})
                station_coverage_file = coverage_config.get('station_coverage_file', 'station_coverage.json')
                
                # StationCoverageCalculatorを読み込み
                coverage_path = self.base_dir / "processed" / station_coverage_file
                if coverage_path.exists():
                    self.station_coverage_calculator = StationCoverageCalculator.load(str(coverage_path))
                    print(f"★ 事前計算カバレッジ読み込み完了: {coverage_path}")
                else:
                    # ファイルがない場合は事前計算を実行
                    print(f"警告: {coverage_path} が見つかりません。事前計算を実行します...")
                    self.station_coverage_calculator = StationCoverageCalculator()
                    self.station_coverage_calculator.compute_coverage(
                        ambulance_data=self.ambulance_data,
                        travel_time_matrix=response_matrix,
                        grid_mapping=self.grid_mapping,
                        verbose=True
                    )
                    # 計算結果を保存
                    self.station_coverage_calculator.save(str(coverage_path))
                    print(f"★ 事前計算カバレッジを保存: {coverage_path}")
                
                # CompactStateEncoderV2を使用
                from .state_encoder_v2 import CompactStateEncoderV2
                self.state_encoder = CompactStateEncoderV2(
                    config=self.config,
                    top_k=self.top_k,
                    travel_time_matrix=response_matrix,
                    grid_mapping=self.grid_mapping,
                    station_coverage_calculator=self.station_coverage_calculator
                )
                
                print("=" * 60)
                print(f"★ コンパクトモード有効: Top-{self.top_k}")
                print(f"  状態次元: {self.state_encoder.state_dim}")
                print(f"  行動次元: {self.action_dim}")
                print(f"  カバレッジ計算: 決定論的方式 (v2)")
                print("=" * 60)
            else:
                # v1: 従来のランダムサンプリング方式
                self.station_coverage_calculator = None  # v1では使用しない
                
                from .state_encoder import CompactStateEncoder
                self.state_encoder = CompactStateEncoder(
                    config=self.config,
                    top_k=self.top_k,
                    travel_time_matrix=response_matrix,
                    grid_mapping=self.grid_mapping
                )
                
                print("=" * 60)
                print(f"★ コンパクトモード有効: Top-{self.top_k}")
                print(f"  状態次元: {self.state_encoder.state_dim}")
                print(f"  行動次元: {self.action_dim}")
                print(f"  カバレッジ計算: ランダムサンプリング方式 (v1)")
                print("=" * 60)
            
            # Top-K救急車のIDを保持するリスト（step()で使用）
            self.current_top_k_ids = []
        else:
            # 従来モード: action_dim = 全救急車数, state_dim = 999
            self.action_dim = len(self.ambulance_data)
            self.station_coverage_calculator = None  # 従来モードでは使用しない
            
            from .state_encoder import StateEncoder
            self.state_encoder = StateEncoder(
                config=self.config,
                max_ambulances=self.action_dim,
                travel_time_matrix=response_matrix,
                grid_mapping=self.grid_mapping
            )
            
            # 従来モードではTop-K IDは使用しない
            self.current_top_k_ids = None
        
        self.state_dim = self.state_encoder.state_dim
        
        print(f"状態空間次元: {self.state_dim}")
        print(f"行動空間次元: {self.action_dim}")
        
        # 統計情報の初期化
        self.episode_stats = self._init_episode_stats()
        
        # RewardDesignerを一度だけ初期化
        from .reward_designer import RewardDesigner
        self.reward_designer = RewardDesigner(self.config)
        
        # DispatchLoggerの初期化
        from .dispatch_logger import DispatchLogger
        self.dispatch_logger = DispatchLogger(enabled=True)
        
        # ServiceTimeGeneratorの初期化
        self._init_service_time_generator()
        
        # 病院選択関連の初期化（ValidationSimulatorと同じ）
        self._classify_hospitals()
        self.use_probabilistic_selection = self.config.get('hospital_selection', {}).get('use_probabilistic', True)
        if self.use_probabilistic_selection:
            self._load_hospital_selection_model()
        
        # ハイブリッドモード設定（従来版）
        self.hybrid_mode = self.config.get('hybrid_mode', {}).get('enabled', False)
        if self.hybrid_mode:
            self.severe_conditions = ['重症', '重篤', '死亡']
            self.mild_conditions = ['軽症', '中等症']
            self.direct_dispatch_count = 0  # 直近隊運用の回数
            self.ppo_dispatch_count = 0     # PPO運用の回数
            print("ハイブリッドモード有効: 重症系は直近隊、軽症系はPPO学習")
        
        # ★★★ ハイブリッドモードv2設定（新機能）★★★
        self.hybrid_v2_enabled = self._is_hybrid_v2_enabled()
        if self.hybrid_v2_enabled:
            self.severe_conditions = ['重症', '重篤', '死亡']
            self.mild_conditions = ['軽症', '中等症']
            self.direct_dispatch_count = 0  # 直近隊運用の回数
            self.ppo_dispatch_count = 0     # PPO運用（フィルタリング後選択）の回数
            self.hybrid_v2_stats = {
                'filtered_candidates_count': [],  # 絞り込み後の候補数
                'severe_cases_count': 0,          # 重症系事案数
                'mild_cases_count': 0             # 軽症系事案数
            }
            hybrid_v2_config = self.config.get('hybrid_v2', {})
            print("=" * 60)
            print("ハイブリッドモードv2有効:")
            print("  - 重症系: 直近隊選択（固定、学習対象外）")
            print("  - 軽症系: 候補絞り込み後、PPOが選択（学習対象）")
            print(f"  - カバレッジ損失閾値: {hybrid_v2_config.get('mild_filtering', {}).get('coverage_loss_threshold', 0.8)}")
            print(f"  - 最低候補数: {hybrid_v2_config.get('mild_filtering', {}).get('min_candidates', 3)}")
            print("=" * 60)
        
    def _row_is_virtual(self, row: pd.Series) -> bool:
        """DataFrame行から仮想フラグを安全に判定（NaNはFalse）。"""
        try:
            value = row.get('is_virtual', False)
        except Exception:
            return False
        # NaN対策
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        # 真偽/文字列の両対応
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ['true', '1', 'yes', 'y']
        if isinstance(value, (int, float)):
            return int(value) == 1
        return False
    
    # ===================================================================
    # ハイブリッドモードv2 メソッド群
    # ===================================================================
    
    def _is_hybrid_v2_enabled(self) -> bool:
        """ハイブリッドモードv2が有効かチェック"""
        return self.config.get('hybrid_v2', {}).get('enabled', False)
    
    def _get_filtered_mask_for_mild(self, base_mask: np.ndarray, severity: str) -> np.ndarray:
        """
        軽症系の候補絞り込み（傷病度考慮運用と同等の条件）
        
        Args:
            base_mask: 基本の行動マスク（利用可能な救急車）
            severity: 傷病度
            
        Returns:
            フィルタリング後の行動マスク
        """
        filtered_mask = np.zeros(self.action_dim, dtype=bool)
        
        request_h3 = self.pending_call.get('h3_index')
        available_ambulances = [amb_id for amb_id in range(self.action_dim) if base_mask[amb_id]]
        
        if not available_ambulances:
            return base_mask
        
        # ハイブリッドv2設定を読み込み
        hybrid_config = self.config.get('hybrid_v2', {}).get('mild_filtering', {})
        
        # 時間制限の取得（設定から、またはデフォルト値）
        use_time_limit = hybrid_config.get('use_time_limit', True)
        if use_time_limit:
            # 設定された時間制限を使用（なければconstantsのデフォルト）
            time_limit = hybrid_config.get('time_limit_seconds', get_severity_time_limit(severity))
        else:
            time_limit = float('inf')
        
        # カバレッジ損失フィルタリング設定
        use_coverage_filter = hybrid_config.get('use_coverage_filter', True)
        coverage_loss_threshold = hybrid_config.get('coverage_loss_threshold', 0.8)
        min_candidates = hybrid_config.get('min_candidates', 3)
        
        candidates = []
        
        for amb_id in available_ambulances:
            amb_state = self.ambulance_states.get(amb_id)
            if not amb_state:
                continue
            
            # 応答時間をチェック
            response_time = self._calculate_travel_time(
                amb_state['current_h3'],
                request_h3
            )
            
            # 時間制限チェック
            if response_time > time_limit:
                continue
            
            # カバレッジ損失をチェック（オプション）
            coverage_loss = 0.0
            if use_coverage_filter:
                try:
                    coverage_loss = self._calculate_coverage_loss(
                        amb_id,
                        available_ambulances,
                        request_h3
                    )
                except Exception:
                    coverage_loss = 0.0
            
            candidates.append({
                'amb_id': amb_id,
                'response_time': response_time,
                'coverage_loss': coverage_loss
            })
        
        # カバレッジ損失でフィルタリング
        if use_coverage_filter and candidates:
            good_candidates = [c for c in candidates if c['coverage_loss'] < coverage_loss_threshold]
            
            # 最低候補数を確保
            if len(good_candidates) < min_candidates:
                remaining = sorted(
                    [c for c in candidates if c not in good_candidates],
                    key=lambda x: x['coverage_loss']
                )
                good_candidates.extend(remaining[:min_candidates - len(good_candidates)])
            
            candidates = good_candidates
        
        # フィルタリング後のマスクを作成
        for c in candidates:
            filtered_mask[c['amb_id']] = True
        
        # 候補がない場合は元のマスクを返す
        if not filtered_mask.any():
            return base_mask
        
        # 統計記録（ハイブリッドv2が有効な場合）
        if hasattr(self, 'hybrid_v2_stats'):
            self.hybrid_v2_stats['filtered_candidates_count'].append(filtered_mask.sum())
        
        return filtered_mask

    def _setup_severity_mapping(self):
        """傷病度マッピングの設定"""
        self.severity_to_category = {}
        self.severity_weights = {}
        
        for category, info in self.config['severity']['categories'].items():
            for condition in info['conditions']:
                self.severity_to_category[condition] = category
                self.severity_weights[condition] = info['reward_weight']
        
        print("傷病度設定:")
        for category, info in self.config['severity']['categories'].items():
            conditions = ', '.join(info['conditions'])
            weight = info['reward_weight']
            print(f"  {category}: {conditions} (重み: {weight})")
    
    def _init_service_time_generator(self):
        """ServiceTimeGeneratorの初期化（ValidationSimulatorと同じロジック）"""
        # サービス時間パラメータファイルの検索（階層的パラメータを優先）
        # 現在のプロジェクトディレクトリからの絶対パスを使用
        data_dir = CURRENT_PROJECT_DIR / "data" / "tokyo"
        hierarchical_params_path = data_dir / "service_time_analysis" / "lognormal_parameters_hierarchical.json"
        standard_params_path = data_dir / "service_time_analysis" / "lognormal_parameters.json"
        
        # ValidationSimulatorと同じ初期化ロジック
        if USE_ENHANCED_GENERATOR and hierarchical_params_path.exists():
            print(f"  階層的パラメータを使用してServiceTimeGeneratorEnhancedを初期化")
            print(f"  パラメータファイル: {hierarchical_params_path}")
            try:
                self.service_time_generator = ServiceTimeGeneratorEnhanced(str(hierarchical_params_path))
                print("  ✓ ServiceTimeGeneratorEnhanced初期化成功")
            except Exception as e:
                print(f"  ❌ ServiceTimeGeneratorEnhanced初期化失敗: {e}")
                print(f"  標準パラメータで再試行します")
                if standard_params_path.exists():
                    self.service_time_generator = ServiceTimeGenerator(str(standard_params_path))
                    print("  ✓ ServiceTimeGenerator（標準版）初期化成功")
                else:
                    self.service_time_generator = None
        elif standard_params_path.exists():
            print(f"  標準パラメータを使用してServiceTimeGeneratorを初期化")
            print(f"  パラメータファイル: {standard_params_path}")
            try:
                self.service_time_generator = ServiceTimeGenerator(str(standard_params_path))
                print("  ✓ ServiceTimeGenerator初期化成功")
            except Exception as e:
                print(f"  ❌ ServiceTimeGenerator初期化失敗: {e}")
                self.service_time_generator = None
        else:
            print("  ❌ サービス時間パラメータファイルが見つかりません")
            print("  フォールバック処理を使用します")
            self.service_time_generator = None
    
    def _classify_hospitals(self):
        """病院を3次救急とそれ以外に分類（ValidationSimulatorと同じロジック）"""
        self.tertiary_hospitals = set()  # 3次救急医療機関
        self.secondary_primary_hospitals = set()  # 2次以下の医療機関
        
        if not hasattr(self, 'hospital_data') or self.hospital_data is None:
            print("警告: 病院データが提供されていません。病院分類をスキップします。")
            return
        
        print("病院を救急医療機関レベル別に分類中...")
        
        # H3インデックスを計算して病院データに追加
        self.hospital_data = self.hospital_data.copy()
        self.hospital_data['h3_index'] = self.hospital_data.apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], 9)
            if pd.notna(row['latitude']) and pd.notna(row['longitude']) else None,
            axis=1
        )
        
        # genre_codeに基づいて分類
        if 'genre_code' in self.hospital_data.columns:
            tertiary_hospitals_df = self.hospital_data[
                (self.hospital_data['genre_code'] == 1) & 
                (self.hospital_data['h3_index'].notna())
            ]
            
            secondary_primary_hospitals_df = self.hospital_data[
                (self.hospital_data['genre_code'] == 2) & 
                (self.hospital_data['h3_index'].notna())
            ]
            
            # H3インデックスをセットに変換
            self.tertiary_hospitals = set(tertiary_hospitals_df['h3_index'].tolist())
            self.secondary_primary_hospitals = set(secondary_primary_hospitals_df['h3_index'].tolist())
            
            # grid_mappingに存在しない病院を除外
            self.tertiary_hospitals = {h3_idx for h3_idx in self.tertiary_hospitals if h3_idx in self.grid_mapping}
            self.secondary_primary_hospitals = {h3_idx for h3_idx in self.secondary_primary_hospitals if h3_idx in self.grid_mapping}
            
            print(f"分類結果:")
            print(f"  3次救急医療機関: {len(self.tertiary_hospitals)}件")
            print(f"  2次以下医療機関: {len(self.secondary_primary_hospitals)}件")
            
            # 分類されなかった病院を2次以下に追加
            all_hospital_h3 = set(self.hospital_data[self.hospital_data['h3_index'].notna()]['h3_index'].tolist())
            all_hospital_h3 = {h3_idx for h3_idx in all_hospital_h3 if h3_idx in self.grid_mapping}
            unclassified = all_hospital_h3 - self.tertiary_hospitals - self.secondary_primary_hospitals
            if unclassified:
                print(f"  未分類病院（2次以下に追加）: {len(unclassified)}件")
                self.secondary_primary_hospitals.update(unclassified)
        else:
            print("警告: hospital_dataに'genre_code'カラムが見つかりません。全ての病院を2次以下として扱います。")
            all_hospital_h3 = set(self.hospital_data[self.hospital_data['h3_index'].notna()]['h3_index'].tolist())
            self.secondary_primary_hospitals = {h3_idx for h3_idx in all_hospital_h3 if h3_idx in self.grid_mapping}
    
    def _load_hospital_selection_model(self):
        """確率的病院選択モデルを読み込む（ValidationSimulatorと同じロジック）"""
        # 現在のプロジェクトディレクトリからの絶対パスを使用
        data_dir = CURRENT_PROJECT_DIR / "data" / "tokyo"
        model_path = data_dir / 'processed' / 'hospital_selection_model_revised.pkl'
        
        try:
            with open(model_path, 'rb') as f:
                main_model = pickle.load(f)
                self.hospital_selection_model = main_model['selection_probabilities']
                self.static_fallback_model = main_model.get('static_fallback_model', {})
                self.model_hospital_master = pd.DataFrame(main_model['hospital_master'])
                self.model_h3_centers = main_model['h3_centers']
            
            print(f"改訂版の確率的病院選択モデルを読み込みました:")
            print(f"  実績ベースの条件数: {len(self.hospital_selection_model)}")
            if self.static_fallback_model:
                print(f"  静的フォールバックモデルの傷病度: {list(self.static_fallback_model.keys())}")
        except FileNotFoundError as e:
            print(f"警告: 確率モデルファイルが見つかりません: {e}")
            print("デフォルトの最寄り病院選択を使用します。")
            self.use_probabilistic_selection = False
            self.hospital_selection_model = None
            self.static_fallback_model = None
    
    def _load_base_data(self):
        """基本データの読み込み（修正版：ValidationSimulatorと同じフィルタリング）"""
        print("\n基本データ読み込み中...")
        
        # 救急署データ
        firestation_path = self.base_dir / "import/amb_place_master.csv"
        ambulance_data_full = pd.read_csv(firestation_path, encoding='utf-8')
        ambulance_data_full = ambulance_data_full[ambulance_data_full['special_flag'] == 1]
        
        print(f"  元データ: {len(ambulance_data_full)}台")
        
        # ★★★ 修正1: 常に「救急隊なし」を除外 ★★★
        if 'team_name' in ambulance_data_full.columns:
            before_exclusion = len(ambulance_data_full)
            
            # ValidationSimulatorと同じフィルタリング
            team_mask = (ambulance_data_full['team_name'] != '救急隊なし')
            ambulance_data_full = ambulance_data_full[team_mask].copy()
            
            excluded_count = before_exclusion - len(ambulance_data_full)
            print(f"  「救急隊なし」除外: {before_exclusion}台 → {len(ambulance_data_full)}台 (除外: {excluded_count}台)")
        
        # ★★★ 修正2: デイタイム救急の除外（オプション）★★★
        # config.yamlで制御できるようにする
        exclude_daytime = self.config.get('data', {}).get('exclude_daytime_ambulances', True)
        
        if exclude_daytime and 'team_name' in ambulance_data_full.columns:
            before_daytime = len(ambulance_data_full)
            
            daytime_mask = ~ambulance_data_full['team_name'].str.contains('デイタイム', na=False)
            ambulance_data_full = ambulance_data_full[daytime_mask].copy()
            
            excluded_daytime = before_daytime - len(ambulance_data_full)
            print(f"  「デイタイム救急」除外: {before_daytime}台 → {len(ambulance_data_full)}台 (除外: {excluded_daytime}台)")
        
        print(f"  フィルタリング後: {len(ambulance_data_full)}台")
        
        # エリア制限フィルタリングの設定確認
        area_restriction = self.config.get('data', {}).get('area_restriction', {})
        
        if area_restriction.get('enabled', False):
            section_code = area_restriction.get('section_code')
            area_name = area_restriction.get('area_name', '指定エリア')
            
            # section_codeがnullまたはNoneの場合は全方面を使用（東京23区全域など）
            if section_code is None or section_code == 'null':
                print(f"  {area_name}（全方面）を使用")
                self.ambulance_data = ambulance_data_full
                
            elif section_code in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                # 指定方面の救急隊に限定
                before_filter = len(ambulance_data_full)
                section_filtered = ambulance_data_full[ambulance_data_full['section'] == section_code].copy()
                
                self.ambulance_data = section_filtered
                print(f"  {area_name}フィルタ適用: {before_filter}台 → {len(self.ambulance_data)}台")
                
                if len(self.ambulance_data) == 0:
                    print(f"  警告: {area_name}の救急車が見つかりません。全体を使用します。")
                    self.ambulance_data = ambulance_data_full
            else:
                # その他の場合は全体を使用
                self.ambulance_data = ambulance_data_full
        else:
            # ★★★ 修正3: エリア制限なしでもフィルタリング済みデータを使用 ★★★
            self.ambulance_data = ambulance_data_full
        
        print(f"  最終救急車数: {len(self.ambulance_data)}台")
        
        # ★★★ 修正4: ValidationSimulatorとの一致確認 ★★★
        print(f"\n  ✓ ValidationSimulatorとの一致確認:")
        print(f"    - 「救急隊なし」: 除外済み")
        if exclude_daytime:
            print(f"    - 「デイタイム救急」: 除外済み")
        else:
            print(f"    - 「デイタイム救急」: 含む")
        print(f"    - 最終台数: {len(self.ambulance_data)}台")
        print(f"    - この台数がValidationSimulatorと一致する必要があります")
        
        # ★★★ 仮想救急車の作成（学習モード時）★★★
        if self.mode == 'train':
            self.ambulance_data = self._create_virtual_ambulances_if_needed(self.ambulance_data)
            print(f"  最終救急車数（仮想含む）: {len(self.ambulance_data)}台")
        
        # 病院データ（方面に関係なく全体を使用）
        hospital_path = self.base_dir / "import/hospital_master.csv"
        self.hospital_data = pd.read_csv(hospital_path, encoding='utf-8')
        # ValidationSimulatorと同じカラム名変更
        self.hospital_data = self.hospital_data.rename(columns={
            'hospital_latitude': 'latitude', 
            'hospital_longitude': 'longitude'
        })
        print(f"  病院数: {len(self.hospital_data)}")
        
        # グリッドマッピング
        grid_mapping_path = self.base_dir / "processed/grid_mapping_res9.json"
        with open(grid_mapping_path, 'r', encoding='utf-8') as f:
            self.grid_mapping = json.load(f)
        print(f"  H3グリッド数: {len(self.grid_mapping)}")
        
        # 移動時間行列（軽量版 - 学習用）
        self.travel_time_matrices = {}
        calibration_dir = self.base_dir / "calibration2"
        for phase in ['response', 'transport', 'return']:
            matrix_path = calibration_dir / f"linear_calibrated_{phase}.npy"
            if matrix_path.exists():
                self.travel_time_matrices[phase] = np.load(matrix_path)
        
        # 距離行列
        distance_matrix_path = self.base_dir / "processed/travel_distance_matrix_res9.npy"
        self.travel_distance_matrix = np.load(distance_matrix_path)
        
    def _calculate_state_dim(self) -> int:
        """状態空間の次元を計算（フォールバック用）"""
        # StateEncoderが既に次元を計算しているので、そこから取得するだけ
        if hasattr(self, 'state_encoder'):
            return self.state_encoder.state_dim
        else:
            # 古いフォールバックロジック
            actual_ambulance_count = self.action_dim if hasattr(self, 'action_dim') else len(self.ambulance_data)
            ambulance_features = actual_ambulance_count * 4
            incident_features = 10
            temporal_features = 8
            spatial_features = 20
            total = ambulance_features + incident_features + temporal_features + spatial_features
            print(f"  状態空間次元: 救急車{actual_ambulance_count}台 × 4 + その他{incident_features + temporal_features + spatial_features} = {total}")
            return total


    def set_mode(self, mode: str):
        """
        環境のモードを切り替える（トレーニング/評価）
        
        Args:
            mode: "train" または "eval"
            
        Raises:
            ValueError: 無効なモードが指定された場合
        """
        if mode not in ["train", "eval"]:
            raise ValueError(f"無効なモード: {mode}. 'train' または 'eval' を指定してください。")
        
        old_mode = self.mode
        self.mode = mode
        
        # モード切り替え時はログフラグをリセット（期間情報を再表示するため）
        if old_mode != mode:
            self._first_period_logged = False
            print(f"環境モード切り替え: {old_mode} → {mode}")

    
    def reset(self, period_index: Optional[int] = None) -> np.ndarray:
        """
        環境のリセット
        
        Returns:
            初期観測
        """
        # 期間の選択
        if self.mode == "train":
            periods = self.config['data']['train_periods']
        else:
            periods = self.config['data']['eval_periods']
        
        if period_index is None:
            period_index = np.random.randint(len(periods))
        
        self.current_period_idx = period_index
        period = periods[period_index]
        
        # エピソード開始情報は最初の期間のみ表示
        if not self._first_period_logged:
            print(f"\nエピソード開始: {period['start_date']} - {period['end_date']}")
            self._first_period_logged = True
        
        # エピソードカウンタをインクリメント
        self._episode_count += 1
        
        # シミュレータの初期化
        self._init_simulator_for_period(period)
        
        # エピソード統計のリセット
        self.episode_stats = self._init_episode_stats()
        
        # 対応不能事案管理の初期化
        self.unhandled_calls = []  # 対応不能になった事案のリスト
        self.call_start_times = {}  # 事案の発生時刻記録
        
        # 最初の事案を設定（重要！）
        if len(self.current_episode_calls) > 0:
            self.episode_step = 0
            self.pending_call = self.current_episode_calls[0]
            self.call_start_times[self.pending_call['id']] = self.episode_step
        else:
            print("警告: エピソードに事案がありません")
            self.pending_call = None
        
        # 初期観測を返す
        return self._get_observation()
    
    def _init_simulator_for_period(self, period: Dict):
        """指定期間用のシミュレータを初期化"""
        # 救急事案データの読み込み
        calls_df = self._load_calls_for_period(period)
        
        # エピソード用の事案を準備
        self.current_episode_calls = self._prepare_episode_calls(calls_df)
        
        # max_steps_per_episodeの設定（configの値を優先）
        config_max_steps = self.config.get('data', {}).get('max_steps_per_episode') or \
                          self.config.get('max_steps_per_episode')
        
        if config_max_steps:
            # configで指定されている場合、事案数との小さい方を使用
            self.max_steps_per_episode = min(config_max_steps, len(self.current_episode_calls))
        else:
            # configで指定されていない場合、事案数を使用
            self.max_steps_per_episode = len(self.current_episode_calls)
        
        print(f"読み込まれた事案数: {len(self.current_episode_calls)}")
        if config_max_steps:
            print(f"最大ステップ数: {self.max_steps_per_episode} (config設定: {config_max_steps})")
        
        # 救急車状態の初期化
        self._init_ambulance_states()
        
        # エピソードカウンタ初期化（重要！）
        self.episode_step = 0
        self.pending_call = None
        
    def _load_calls_for_period(self, period: Dict) -> pd.DataFrame:
        """
        指定期間の救急事案データを読み込み（最適化版）
        キャッシュからデータを取得するため高速
        """
        start_date = str(period['start_date'])
        end_date = str(period['end_date'])
        
        # エリア制限の設定確認
        area_restriction = self.config.get('data', {}).get('area_restriction', {})
        area_filter = None
        if area_restriction.get('enabled', False):
            area_filter = area_restriction.get('districts', [])
        
        # 最初の期間のみ詳細情報を表示
        if not self._first_period_logged:
            area_name = area_restriction.get('area_name', 'エリア制限')
            area_info = f" ({area_name}: {', '.join(area_filter)})" if area_filter else ""
            print(f"期間データ取得中: {start_date} - {end_date}{area_info}")
        
        # キャッシュから高速取得（エリアフィルタ付き）
        filtered_df = self.data_cache.get_period_data(start_date, end_date, area_filter)
        
        if not self._first_period_logged:
            print(f"期間内の事案数: {len(filtered_df)}件")
        
        # 必要なカラムの存在確認
        required_columns = ['救急事案番号キー', 'Y_CODE', 'X_CODE', '収容所見程度', '出場年月日時分']
        missing_columns = [col for col in required_columns if col not in filtered_df.columns]
        if missing_columns:
            print(f"警告: 必要なカラムが不足: {missing_columns}")
            return pd.DataFrame()
        
        if not self._first_period_logged:
            print(f"最終的な事案数: {len(filtered_df)}件")
            
            if len(filtered_df) > 0:
                # 傷病度の分布を表示
                severity_counts = filtered_df['収容所見程度'].value_counts()
                print("傷病度分布:")
                for severity, count in severity_counts.head().items():
                    print(f"  {severity}: {count}件")
            print(f"エピソード長: {self.config['data']['episode_duration_hours']}時間")
        
        return filtered_df
    
    def _prepare_episode_calls(self, calls_df: pd.DataFrame) -> List[Dict]:
        """エピソード用の事案リストを準備"""
        import h3
        import numpy as np
        import pandas as pd
        
        if len(calls_df) == 0:
            print("警告: 事案データが空です")
            return []
        
        episode_calls = []
        
        # エピソード長の設定（時間）
        episode_hours = self.config['data']['episode_duration_hours']
        print(f"エピソード長: {episode_hours}時間")
        
        # 時刻でソート
        calls_df = calls_df.sort_values('出場年月日時分')
        
        # エピソードの開始時刻をランダムに選択
        start_time = calls_df['出場年月日時分'].iloc[0]
        end_time = calls_df['出場年月日時分'].iloc[-1]
        
        # エピソード期間内のデータを選択できる開始時刻の範囲
        max_start_time = end_time - pd.Timedelta(hours=episode_hours)
        
        if start_time >= max_start_time:
            # データが短すぎる場合は全体を使用
            episode_start = start_time
            episode_end = end_time
            print(f"警告: データ期間が短いため、全期間を使用")
        else:
            # ランダムな開始時刻を選択
            time_range = (max_start_time - start_time).total_seconds()
            random_offset = np.random.uniform(0, time_range)
            episode_start = start_time + pd.Timedelta(seconds=random_offset)
            episode_end = episode_start + pd.Timedelta(hours=episode_hours)
        
        # エピソード期間内の事案を抽出
        mask = (calls_df['出場年月日時分'] >= episode_start) & (calls_df['出場年月日時分'] <= episode_end)
        episode_df = calls_df[mask].copy()
        
        # 毎回表示する情報（簡潔版）
        print(f"エピソード期間: {episode_start.strftime('%Y-%m-%d %H:%M')} ～ {episode_end.strftime('%Y-%m-%d %H:%M')}")
        print(f"エピソード内事案数: {len(episode_df)}件")
        
        for _, row in episode_df.iterrows():
            # H3インデックスの計算
            try:
                # 座標の有効性チェック
                lat = float(row['Y_CODE'])
                lng = float(row['X_CODE'])
                
                if -90 <= lat <= 90 and -180 <= lng <= 180:
                    h3_index = h3.latlng_to_cell(lat, lng, 9)
                else:
                    continue  # 無効な座標はスキップ
            except Exception as e:
                continue  # 変換エラーはスキップ
            
            call_info = {
                'id': str(row['救急事案番号キー']),
                'h3_index': h3_index,
                'severity': row.get('収容所見程度', 'その他'),
                'datetime': row['出場年月日時分'],
                'location': (lat, lng)
            }
            episode_calls.append(call_info)
        
        # 時間順にソート
        episode_calls.sort(key=lambda x: x['datetime'])
        
        print(f"有効な事案数: {len(episode_calls)}件")
        
        return episode_calls
    
    def _init_ambulance_states(self):
        """救急車の状態を初期化"""
        self.ambulance_states = {}
        
        # 全救急車数（コンパクトモードでもaction_dimではなく実際の救急車数を使用）
        total_ambulances = len(self.ambulance_data)
        
        print(f"  救急車データから初期化開始: {total_ambulances}台のデータ")
        
        # DataFrameのindexではなく、0から始まる連続した番号を使用
        # 注意: コンパクトモードでもaction_dimではなく全救急車を初期化する
        for amb_id, (_, row) in enumerate(self.ambulance_data.iterrows()):
            
            try:
                # 座標の検証
                lat = float(row['latitude'])
                lng = float(row['longitude'])
                
                if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                    print(f"    ⚠️ 救急車{amb_id}: 無効な座標 lat={lat}, lng={lng}")
                    continue
                
                h3_index = h3.latlng_to_cell(lat, lng, 9)
                
                # 表示名（優先: team_name → name → フォールバック）
                # 仮想隊は一律でvirtual_team_{id}、実隊はCSVのteam_name
                is_virtual_row = self._row_is_virtual(row)
                if is_virtual_row:
                    display_name = f"virtual_team_{amb_id}"
                else:
                    try:
                        display_name = row.get('team_name') if pd.notna(row.get('team_name')) else None
                    except Exception:
                        display_name = None
                if not display_name:
                    try:
                        display_name = row.get('name') if pd.notna(row.get('name')) else None
                    except Exception:
                        display_name = None
                if not display_name:
                    display_name = f"救急車{amb_id}"

                self.ambulance_states[amb_id] = {
                    'id': f"amb_{amb_id}",
                    'name': display_name,
                    'station_h3': h3_index,
                    'current_h3': h3_index,
                    'status': 'available',
                    'calls_today': 0,
                    'last_dispatch_time': None
                }
                
            except Exception as e:
                print(f"    ❌ 救急車{amb_id}の初期化でエラー: {e}")
                print(f"       データ: lat={row.get('latitude')}, lng={row.get('longitude')}")
                continue
        
        print(f"  救急車状態初期化完了: {len(self.ambulance_states)}台 (利用可能: {len(self.ambulance_states)}台)")
        
        # 実際の救急車と仮想救急車の数を確認
        real_count = 0
        virtual_count = 0
        for amb_id in range(len(self.ambulance_states)):
            if amb_id < len(self.ambulance_data):
                is_virtual = self._row_is_virtual(self.ambulance_data.iloc[amb_id])
                if is_virtual:
                    virtual_count += 1
                else:
                    real_count += 1
            else:
                virtual_count += 1
        
        print(f"  実際の救急車: {real_count}台, 仮想救急車: {virtual_count}台")
        
        # デバッグ: 最初の数台の救急車の詳細を表示
        print("  救急車詳細（最初の5台）:")
        for amb_id in range(min(5, len(self.ambulance_data))):
            row = self.ambulance_data.iloc[amb_id]
            is_virtual = row.get('is_virtual', False)
            team_name = row.get('team_name', 'unknown')
            print(f"    救急車{amb_id}: {team_name} ({'仮想' if is_virtual else '実車'})")
        
        # 初期化直後のマスクチェック
        initial_mask = self.get_action_mask()
        print(f"  初期化直後の利用可能数: {initial_mask.sum()}台")
        
        # デバッグ: 救急車状態の詳細確認
        if initial_mask.sum() == 0:
            print("  ⚠️ 警告: 初期化時に利用可能な救急車が0台です")
            for amb_id, amb_state in self.ambulance_states.items():
                print(f"    救急車{amb_id}: status={amb_state['status']}, h3={amb_state['current_h3']}")
        else:
            print(f"  ✓ 正常: {initial_mask.sum()}台の救急車が利用可能")
    
    def step(self, action: int) -> StepResult:
        """
        環境のステップ実行（ハイブリッドモードv2対応版、コンパクトモード対応）
        
        Args:
            action: 選択された救急車のインデックス
                    コンパクトモード時はTop-K内のインデックス（0-9）
                    従来モード時は救急車ID（0-191）
            
        Returns:
            StepResult: 観測、報酬、終了フラグ、追加情報
        """
        try:
            # 現在の事案を取得（コンパクトモード処理より前に定義）
            current_incident = self.pending_call
            
            # ========== コンパクトモード: actionをTop-K内インデックスとして解釈 ==========
            if self.compact_mode:
                if self.current_top_k_ids and action < len(self.current_top_k_ids):
                    actual_ambulance_id = self.current_top_k_ids[action]
                else:
                    # フォールバック: Top-K IDsがない、または範囲外の場合
                    if self.current_top_k_ids:
                        # Top-K IDsがあるが、actionが範囲外 → Top-1を選択
                        actual_ambulance_id = self.current_top_k_ids[0]
                    else:
                        # 利用可能な救急車がない場合 → 従来の直近隊選択ロジックを使用
                        closest = self._get_closest_ambulance_action(current_incident)
                        actual_ambulance_id = closest if closest is not None else 0
                    # 注意: 利用可能な救急車がない場合は警告を出さない（正常なケース）
            else:
                # 従来モード: actionがそのまま救急車ID
                actual_ambulance_id = action
            
            # ★★★ ハイブリッドモードv2: 重症系は直近隊を強制 ★★★
            if self._is_hybrid_v2_enabled() and current_incident:
                severity = current_incident.get('severity', '')
                
                if is_severe_condition(severity):
                    # 重症系: 直近隊選択を強制（学習対象外）
                    optimal_action = self.get_optimal_action()
                    
                    # コンパクトモードでは、optimal_actionはTop-Kインデックス（0）なので
                    # 実際の救急車IDに変換する必要がある
                    if self.compact_mode:
                        if self.current_top_k_ids and optimal_action is not None and optimal_action < len(self.current_top_k_ids):
                            severe_ambulance_id = self.current_top_k_ids[optimal_action]
                        elif self.current_top_k_ids:
                            severe_ambulance_id = self.current_top_k_ids[0]
                        else:
                            # フォールバック: 従来の直近隊選択ロジック
                            severe_ambulance_id = self._get_closest_ambulance_action(current_incident)
                    else:
                        # 従来モード: optimal_actionがそのまま救急車ID
                        severe_ambulance_id = optimal_action if optimal_action is not None else self._get_closest_ambulance_action(current_incident)
                    
                    self.direct_dispatch_count += 1
                    if hasattr(self, 'hybrid_v2_stats'):
                        self.hybrid_v2_stats['severe_cases_count'] += 1
                    
                    # 直近隊で配車
                    available_before = self._get_available_ambulance_ids()
                    dispatch_result = self._dispatch_ambulance(severe_ambulance_id, available_before)
                    
                    # 報酬は0（学習対象外）
                    reward = 0.0
                    
                    # 統計情報を記録
                    info = {
                        'dispatch_result': dispatch_result,
                        'dispatch_type': 'hybrid_v2_direct_closest',
                        'severity': severity,
                        'response_time': dispatch_result.get('response_time', 0),
                        'skipped_learning': True,  # 学習対象外フラグ
                        'episode_stats': self.episode_stats.copy(),
                        'step': self.episode_step
                    }
                    
                else:
                    # 軽症系: PPOのフィルタリング済み選択を使用（学習対象）
                    self.ppo_dispatch_count += 1
                    if hasattr(self, 'hybrid_v2_stats'):
                        self.hybrid_v2_stats['mild_cases_count'] += 1
                    
                    # PPOの行動を実行（コンパクトモードではactual_ambulance_idを使用）
                    available_before = self._get_available_ambulance_ids()
                    dispatch_result = self._dispatch_ambulance(actual_ambulance_id, available_before)
                    
                    # カバレッジ情報を計算
                    coverage_info = self._calculate_coverage_info()
                    
                    # 報酬計算（学習対象）
                    reward = self._calculate_reward(dispatch_result)
                    
                    # 追加情報
                    info = {
                        'dispatch_result': dispatch_result,
                        'outcome': {
                            'severity': severity,
                            'response_time': dispatch_result.get('response_time', 0)
                        },
                        'coverage_info': coverage_info,
                        'dispatch_type': 'hybrid_v2_ppo_filtered',
                        'severity': severity,
                        'skipped_learning': False,  # 学習対象
                        'episode_stats': self.episode_stats.copy(),
                        'step': self.episode_step
                    }
                
                # ログを記録
                if dispatch_result['success']:
                    self._log_dispatch_action(dispatch_result, self.ambulance_states[dispatch_result['ambulance_id']])
                
                # 統計情報の更新
                self._update_statistics(dispatch_result)
                
                # 次の事案へ進む
                self._advance_to_next_call()
                
                # エピソード終了判定
                done = self._is_episode_done()
                
                # 次の観測を取得
                observation = self._get_observation()
                
                # StepResultオブジェクトを返す
                return StepResult(
                    observation=observation,
                    reward=reward,
                    done=done,
                    info=info
                )
            
            # 従来のハイブリッドモード：重症系は直近隊運用を強制
            if self.hybrid_mode and current_incident:
                severity = current_incident.get('severity', '')
                
                if severity in self.severe_conditions:
                    # 重症系：直近隊運用を実行
                    self.direct_dispatch_count += 1
                    
                    # 直近隊を自動選択
                    closest_action = self._get_closest_ambulance_action(current_incident)
                    
                    # 直近隊で配車
                    available_before = self._get_available_ambulance_ids()
                    dispatch_result = self._dispatch_ambulance(closest_action, available_before)
                    
                    # 報酬は0（学習対象外）
                    reward = 0.0
                    
                    # 統計情報を記録
                    info = {
                        'dispatch_result': dispatch_result,
                        'dispatch_type': 'direct_closest',
                        'severity': severity,
                        'response_time': dispatch_result.get('response_time', 0),
                        'skipped_learning': True,
                        'episode_stats': self.episode_stats.copy(),
                        'step': self.episode_step
                    }
                    
                else:
                    # 軽症系：PPOで学習
                    self.ppo_dispatch_count += 1
                    
                    # PPOの行動を実行（コンパクトモードではactual_ambulance_idを使用）
                    available_before = self._get_available_ambulance_ids()
                    dispatch_result = self._dispatch_ambulance(actual_ambulance_id, available_before)
                    
                    # カバレッジ情報を計算
                    coverage_info = self._calculate_coverage_info()
                    
                    # 報酬計算
                    reward = self._calculate_reward(dispatch_result)
                    
                    # 追加情報
                    info = {
                        'dispatch_result': dispatch_result,
                        'outcome': {
                            'severity': severity,
                            'response_time': dispatch_result.get('response_time', 0)
                        },
                        'coverage_info': coverage_info,
                        'dispatch_type': 'ppo_learning',
                        'severity': severity,
                        'episode_stats': self.episode_stats.copy(),
                        'step': self.episode_step
                    }
            else:
                # 通常モード（既存の処理）
                # デバッグ用: 最適行動との比較を出力
                if hasattr(self, 'verbose_logging') and self.verbose_logging:
                    optimal_action = self.get_optimal_action()
                    # コンパクトモードではoptimal_actionはTop-Kインデックス（0）を返す
                    if optimal_action is not None and action != optimal_action:
                        if self.compact_mode:
                            # コンパクトモード: actual_ambulance_idを使用して比較
                            optimal_amb_id = self.current_top_k_ids[0] if self.current_top_k_ids else 0
                        else:
                            optimal_amb_id = optimal_action
                        optimal_time = self._calculate_travel_time(
                            self.ambulance_states[optimal_amb_id]['current_h3'],
                            self.pending_call['h3_index']
                        )
                        actual_time = self._calculate_travel_time(
                            self.ambulance_states[actual_ambulance_id]['current_h3'],
                            self.pending_call['h3_index']
                        )
                        print(f"[選択比較] PPO選択: 救急車{actual_ambulance_id}({actual_time/60:.1f}分) "
                            f"vs 最適: 救急車{optimal_amb_id}({optimal_time/60:.1f}分)")
                
                # 行動の実行（救急車の配車）- コンパクトモードではactual_ambulance_idを使用
                available_before = self._get_available_ambulance_ids()
                dispatch_result = self._dispatch_ambulance(actual_ambulance_id, available_before)
                
                # 報酬の計算
                reward = self._calculate_reward(dispatch_result)
                
                # 追加情報
                info = {
                    'dispatch_result': dispatch_result,
                    'episode_stats': self.episode_stats.copy(),
                    'step': self.episode_step
                }
            
            # ログを記録
            if dispatch_result['success']:
                self._log_dispatch_action(dispatch_result, self.ambulance_states[dispatch_result['ambulance_id']])
            
            # 統計情報の更新
            self._update_statistics(dispatch_result)
            
            # 次の事案へ進む
            self._advance_to_next_call()
            
            # エピソード終了判定
            done = self._is_episode_done()
            
            # 次の観測を取得
            observation = self._get_observation()
            
            # StepResultオブジェクトを返す
            return StepResult(
                observation=observation,
                reward=reward,
                done=done,
                info=info
            )
        except Exception as e:
            print(f"❌ step()メソッドでエラー発生: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_optimal_action(self) -> Optional[int]:
        """
        現在の事案に対して最適な救急車を選択（最近接）
        ValidationSimulatorのfind_closest_available_ambulanceと同じロジック
        
        Returns:
            コンパクトモード: 常に0（Top-1 = 最短移動時間）
            従来モード: 最適な救急車のID、または None
        """
        if self.pending_call is None:
            return None
        
        # ========== コンパクトモード ==========
        if self.compact_mode:
            # Top-1（最短移動時間）が最適
            # action=0 が常に最短移動時間の救急車
            return 0
        
        # ========== 従来モード ==========
        best_action = None
        min_travel_time = float('inf')
        
        # 全ての救急車をチェック
        for amb_id, amb_state in self.ambulance_states.items():
            # 利用可能な救急車のみ対象
            if amb_state['status'] != 'available':
                continue
            
            try:
                # 現在位置から事案発生地点への移動時間を計算
                travel_time = self._calculate_travel_time(
                    amb_state['current_h3'],
                    self.pending_call['h3_index']
                )
                
                # より近い救急車を発見
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    best_action = amb_id
                    
            except Exception as e:
                # エラーが発生した場合はスキップ
                continue
        
        # デバッグ情報の出力（verboseモード時）
        if best_action is not None and hasattr(self, 'verbose_logging') and self.verbose_logging:
            print(f"[最適選択] 救急車{best_action}を選択 (移動時間: {min_travel_time/60:.1f}分)")
        
        return best_action

    def _get_closest_ambulance_action(self, incident):
        """最寄りの救急車を選択するアクション番号を取得"""
        available_ambulances = self.get_action_mask()
        if not available_ambulances.any():
            return 0
        
        min_distance = float('inf')
        closest_idx = 0
        
        for idx, is_available in enumerate(available_ambulances):
            if is_available and idx < len(self.ambulance_states):
                amb_state = self.ambulance_states[idx]
                distance = self._calculate_travel_time(
                    amb_state['current_h3'], 
                    incident['h3_index']
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = idx
        
        return closest_idx

    def _calculate_coverage_info(self):
        """カバレッジ情報を計算"""
        # 各地域の空き救急車までの平均距離を計算
        coverage_scores = []
        high_risk_scores = []
        
        # 簡易的なカバレッジ計算（利用可能救急車の割合）
        available_count = sum(1 for amb in self.ambulance_states.values() 
                             if amb['status'] == 'available')
        total_count = len(self.ambulance_states)
        
        if total_count > 0:
            overall_coverage = available_count / total_count
            coverage_scores.append(overall_coverage)
            
            # 高リスク地域の判定（簡易版：重症系事案が多い地域を想定）
            # 実際の実装では、過去の重症系事案データから高リスク地域を特定
            high_risk_coverage = overall_coverage  # 簡略化
        
        return {
            'overall_coverage': np.mean(coverage_scores) if coverage_scores else 0.0,
            'high_risk_area_coverage': high_risk_coverage if 'high_risk_coverage' in locals() else 0.0,
            'min_coverage': min(coverage_scores) if coverage_scores else 0.0
        }

    def _get_available_ambulance_ids(self) -> List[int]:
        """現在利用可能な救急車IDのリストを取得"""
        return [
            amb_id for amb_id, state in self.ambulance_states.items()
            if state.get('status') == 'available'
        ]
    
    def _calculate_coverage_loss(self,
                                 selected_ambulance_id: int,
                                 available_ambulances_before: Optional[List[int]],
                                 request_h3: Optional[str]) -> float:
        """選択した救急車によるカバレッジ損失を計算（v2: 事前計算方式対応）"""
        if self.reward_designer.mode != 'coverage_aware':
            return 0.0
        if selected_ambulance_id not in self.ambulance_states:
            return 0.0
        if not available_ambulances_before:
            return 0.0
        
        selected_state = self.ambulance_states[selected_ambulance_id]
        station_h3 = selected_state.get('station_h3') or selected_state.get('current_h3')
        if not station_h3:
            return 0.0
        
        # ★★★ v13: 事前計算方式が利用可能な場合はそちらを使用 ★★★
        if self.station_coverage_calculator is not None:
            return self._calculate_coverage_loss_v2(
                selected_ambulance_id,
                available_ambulances_before,
                station_h3
            )
        
        # ★★★ 以下は従来のランダムサンプリング方式（フォールバック） ★★★
        coverage_config = self.config.get('reward', {}).get('core', {}).get('mild_params', {})
        sample_points = coverage_config.get('sample_points', 20)
        sample_radius = coverage_config.get('sample_radius', 2)
        weight_6min = coverage_config.get('coverage_6min_weight', 0.5)
        weight_13min = coverage_config.get('coverage_13min_weight', 0.5)
        thresholds = self.config.get('severity', {}).get('thresholds', {})
        time_threshold_6 = thresholds.get('golden_time', 360)
        time_threshold_13 = thresholds.get('standard_time', 780)
        
        remaining_ambulances = [
            amb_id for amb_id in available_ambulances_before
            if amb_id != selected_ambulance_id and amb_id in self.ambulance_states
        ]
        if not remaining_ambulances:
            return 1.0
        
        samples = self._get_coverage_sample_points_for_loss(station_h3, sample_points, sample_radius)
        if not samples:
            return self._simple_coverage_loss(station_h3, remaining_ambulances)
        
        coverage_6min_before = coverage_6min_after = 0
        coverage_13min_before = coverage_13min_after = 0
        
        for point_h3 in samples:
            min_before = self._get_min_response_time_for_coverage(point_h3, available_ambulances_before)
            min_after = self._get_min_response_time_for_coverage(point_h3, remaining_ambulances)
            
            if min_before <= time_threshold_6:
                coverage_6min_before += 1
            if min_before <= time_threshold_13:
                coverage_13min_before += 1
            
            if min_after <= time_threshold_6:
                coverage_6min_after += 1
            if min_after <= time_threshold_13:
                coverage_13min_after += 1
        
        total_points = len(samples)
        if total_points == 0:
            return 0.0
        
        loss_6 = (coverage_6min_before - coverage_6min_after) / total_points
        loss_13 = (coverage_13min_before - coverage_13min_after) / total_points
        combined_loss = loss_6 * weight_6min + loss_13 * weight_13min
        return max(0.0, min(1.0, combined_loss))
    
    def _calculate_coverage_loss_v2(self,
                                    selected_ambulance_id: int,
                                    available_ambulances_before: List[int],
                                    station_h3: str) -> float:
        """
        事前計算方式によるカバレッジ損失計算（v2）
        
        Args:
            selected_ambulance_id: 選択された救急車ID
            available_ambulances_before: 選択前に利用可能だった救急車IDリスト
            station_h3: 選択された救急車の署所H3
            
        Returns:
            float: カバレッジ損失（0-1の範囲、L6とL13の重み付け平均）
        """
        if self.station_coverage_calculator is None:
            return 0.5  # フォールバック
        
        # 利用可能な署所H3の集合を構築
        available_station_h3s = set()
        ambulances_per_station = {}
        
        for amb_id in available_ambulances_before:
            if amb_id not in self.ambulance_states:
                continue
            amb_state = self.ambulance_states[amb_id]
            amb_station_h3 = amb_state.get('station_h3')
            if amb_station_h3:
                available_station_h3s.add(amb_station_h3)
                ambulances_per_station[amb_station_h3] = \
                    ambulances_per_station.get(amb_station_h3, 0) + 1
        
        # StationCoverageCalculatorを使用してカバレッジ損失を計算
        try:
            L6, L13 = self.station_coverage_calculator.calculate_coverage_loss(
                departing_station_h3=station_h3,
                available_station_h3s=available_station_h3s,
                ambulances_per_station=ambulances_per_station
            )
            
            # L6とL13の重み付け平均を返す
            coverage_config = self.config.get('reward', {}).get('unified', {})
            weight_6min = coverage_config.get('L6_weight', 0.5)
            weight_13min = coverage_config.get('L13_weight', 0.5)
            
            combined_loss = L6 * weight_6min + L13 * weight_13min
            return max(0.0, min(1.0, combined_loss))
            
        except Exception as e:
            print(f"警告: カバレッジ損失計算エラー (v2): {e}")
            return 0.5
    
    def _get_coverage_sample_points_for_loss(self, center_h3: str, sample_size: int, radius: int) -> List[str]:
        """カバレッジ計算用のサンプルポイントを取得"""
        if not center_h3:
            return []
        try:
            candidates = h3.grid_disk(center_h3, radius)
        except Exception:
            return []
        
        valid = [cell for cell in candidates if cell in self.grid_mapping]
        if len(valid) <= sample_size:
            return valid
        return random.sample(valid, sample_size)
    
    def _get_min_response_time_for_coverage(self, target_h3: str, ambulance_ids: List[int]) -> float:
        """指定地点への最小応答時間を計算"""
        if not ambulance_ids:
            return float('inf')
        
        min_time = float('inf')
        for amb_id in ambulance_ids:
            state = self.ambulance_states.get(amb_id)
            if not state:
                continue
            travel_time = self._calculate_travel_time(state.get('current_h3'), target_h3)
            if travel_time < min_time:
                min_time = travel_time
        return min_time
    
    def _simple_coverage_loss(self, station_h3: str, remaining_ambulances: List[int]) -> float:
        """簡易的なカバレッジ損失（近隣救急車数ベース）"""
        threshold = 600  # 10分
        nearby = 0
        for amb_id in remaining_ambulances:
            state = self.ambulance_states.get(amb_id)
            if not state:
                continue
            travel_time = self._calculate_travel_time(state.get('current_h3'), station_h3)
            if travel_time <= threshold:
                nearby += 1
        return 1.0 / (nearby + 1)
    
    def _calculate_coverage_component_for_stats(self, severity: str, coverage_loss: float) -> float:
        """統計記録用のcoverage_componentを計算"""
        if not self.reward_designer.coverage_aware_params:
            return 0.0
        
        from .reward_designer import severity_to_category
        params = self.reward_designer.coverage_aware_params
        category = severity_to_category(severity)
        coverage_loss = max(0.0, min(1.0, coverage_loss or 0.0))
        
        if category == 'critical':
            severe = params['severe']
            coverage_component = -coverage_loss * severe.get('coverage_weight', 0.0) * abs(severe.get('coverage_loss_penalty_scale', 0.0))
            return coverage_component
        
        mild = params['mild']
        coverage_weight = mild.get('coverage_weight', 0.4)
        coverage_scale = mild.get('coverage_loss_penalty_scale', -100.0)
        coverage_component = coverage_loss * coverage_scale * coverage_weight
        return coverage_component

    def is_high_risk_area(self, area):
        """高リスク地域の判定（簡易版）"""
        # 実際の実装では、過去の重症系事案データから判定
        # ここでは簡易的にランダムで判定
        return np.random.random() < 0.3  # 30%の確率で高リスク地域

    def get_available_count_in_area(self, area):
        """指定エリアの利用可能救急車数を取得（簡易版）"""
        # 実際の実装では、エリア内の救急車をカウント
        # ここでは全体の利用可能数を返す
        return sum(1 for amb in self.ambulance_states.values() 
                  if amb['status'] == 'available')

    def get_total_count_in_area(self, area):
        """指定エリアの総救急車数を取得（簡易版）"""
        # 実際の実装では、エリア内の救急車をカウント
        # ここでは全体の総数を返す
        return len(self.ambulance_states)

    @property
    def areas(self):
        """エリアリストを取得（簡易版）"""
        # 実際の実装では、地理的エリアのリストを返す
        # ここでは簡易的に1つのエリアを返す
        return ['default_area']

    
    def _dispatch_ambulance(self, action: int, available_snapshot: Optional[List[int]] = None) -> Dict:
        """救急車を配車"""
        if self.pending_call is None:
            return {'success': False, 'reason': 'no_pending_call'}
        
        # 行動の妥当性チェック
        if action >= len(self.ambulance_states):
            return {'success': False, 'reason': 'invalid_action'}
        
        if available_snapshot is None:
            available_snapshot = self._get_available_ambulance_ids()
        
        amb_state = self.ambulance_states[action]
        
        # 利用可能性チェック
        if amb_state['status'] != 'available':
            return {'success': False, 'reason': 'ambulance_busy'}
        
        # 移動時間の計算（修正版）
        travel_time = self._calculate_travel_time(
            amb_state['current_h3'],
            self.pending_call['h3_index']
        )
        
        # 配車実行
        amb_state['status'] = 'dispatched'
        amb_state['calls_today'] += 1
        amb_state['last_dispatch_time'] = self.episode_step
        amb_state['current_severity'] = self.pending_call['severity']  # 傷病度を記録
        
        # ValidationSimulatorと同じ活動時間計算
        completion_time = self._calculate_ambulance_completion_time(
            action, self.pending_call, travel_time
        )
        amb_state['call_completion_time'] = completion_time
        
        result = {
            'success': True,
            'ambulance_id': action,
            'call_id': self.pending_call['id'],
            'severity': self.pending_call['severity'],
            'response_time': travel_time,
            'response_time_minutes': travel_time / 60.0,
            'estimated_completion_time': completion_time,
            'matched_teacher': self.current_matched_teacher,
            'available_ambulances_before': list(available_snapshot) if available_snapshot else [],
            'request_h3': self.pending_call['h3_index']
        }
        
        return result
    
    def _log_dispatch_action(self, dispatch_result: Dict, ambulance_state: Dict):
        """配車アクションのログを記録"""
        if not hasattr(self, 'dispatch_logger') or not self.dispatch_logger.enabled:
            return
        
        # 最適救急車を取得
        optimal_ambulance_id = self.get_optimal_action()
        optimal_response_time = None
        if optimal_ambulance_id is not None:
            optimal_response_time = self._calculate_travel_time(
                self.ambulance_states[optimal_ambulance_id]['current_h3'],
                self.pending_call['h3_index']
            ) / 60.0
        
        # 利用可能救急車数とアクションマスク情報
        available_count = sum(1 for amb in self.ambulance_states.values() 
                            if amb['status'] == 'available')
        total_count = len(self.ambulance_states)
        action_mask = self.get_action_mask()
        valid_action_count = action_mask.sum()
        
        # エピソード平均報酬
        episode_reward_avg = 0.0
        if hasattr(self, 'episode_stats') and self.episode_stats['total_dispatches'] > 0:
            # 簡易的な平均報酬計算
            episode_reward_avg = sum(self.episode_stats.get('rewards', [0])) / max(1, len(self.episode_stats.get('rewards', [1])))
        
        # 救急車情報を準備
        ambulance_id = dispatch_result['ambulance_id']
        
        # シンプルかつ安全な判定: DataFrameから直接確認（NaNはFalse）
        is_virtual = False
        try:
            if ambulance_id < len(self.ambulance_data):
                is_virtual = self._row_is_virtual(self.ambulance_data.iloc[ambulance_id])
        except Exception:
            is_virtual = False
        
        # 表示名（デフォルトは状態キャッシュのname、なければデータフレーム由来）
        display_name = ambulance_state.get('name')
        try:
            if not display_name and ambulance_id < len(self.ambulance_data):
                row = self.ambulance_data.iloc[ambulance_id]
                # 仮想は一律virtual_team、実隊はteam_name
                if self._row_is_virtual(row):
                    display_name = f"virtual_team_{ambulance_id}"
                else:
                    display_name = row.get('team_name') or row.get('name') or f"救急車{ambulance_id}"
        except Exception:
            display_name = f"救急車{ambulance_id}"

        ambulance_info = {
            'station_h3': ambulance_state.get('station_h3', 'unknown'),
            'is_virtual': is_virtual,
            'response_time_minutes': dispatch_result['response_time_minutes'],
            'name': display_name
        }

        # この時点で報酬を計算
        if dispatch_result['success']:
            reward = self._calculate_reward(dispatch_result)
        else:
            reward = 0.0
        
        # ログを記録
        self.dispatch_logger.log_dispatch(
            episode=self._episode_count,
            step=self.episode_step,
            call_info=self.pending_call,
            selected_ambulance_id=dispatch_result['ambulance_id'],
            ambulance_info=ambulance_info,
            response_time_minutes=dispatch_result['response_time_minutes'],
            available_count=available_count,
            total_count=total_count,
            action_mask_valid_count=valid_action_count,
            optimal_ambulance_id=optimal_ambulance_id,
            optimal_response_time=optimal_response_time,
            teacher_match=dispatch_result.get('matched_teacher', False),
            reward=reward,  # 報酬は後で計算される
            episode_reward_avg=episode_reward_avg
        )
    
    def _calculate_ambulance_completion_time(self, ambulance_id: int, call: Dict, response_time: float) -> float:
        """救急車の活動完了時間を計算（ValidationSimulator互換）"""
        current_time = self.episode_step  # 現在時刻（分単位）
        severity = call['severity']
        
        # 1. 現場到着時刻 = 現在時刻 + 応答時間
        arrive_scene_time = current_time + (response_time / 60.0)
        
        # 2. 現場活動時間（ServiceTimeGeneratorを使用、call_datetimeも渡す）
        call_datetime = call.get('call_datetime')  # call_datetimeを取得
        
        if self.service_time_generator:
            try:
                # ServiceTimeGeneratorEnhancedの場合はcall_datetimeを渡す
                import inspect
                sig = inspect.signature(self.service_time_generator.generate_time)
                if 'call_datetime' in sig.parameters:
                    # 拡張版の場合
                    on_scene_time = self.service_time_generator.generate_time(
                        severity, 'on_scene_time', call_datetime=call_datetime
                    )
                else:
                    # 従来版の場合
                    on_scene_time = self.service_time_generator.generate_time(severity, 'on_scene_time')
            except Exception as e:
                print(f"🚨 FALLBACK使用: 現場活動時間生成エラー({severity}, on_scene_time): {e}")
                print(f"   正確なサービス時間ではなく推定値を使用しています！")
                # フォールバック: ランダムな現場活動時間
                if severity in ['重篤', '重症']:
                    on_scene_time = np.random.lognormal(np.log(20.0), 0.5)
                elif severity == '中等症':
                    on_scene_time = np.random.lognormal(np.log(15.0), 0.5)
                else:  # 軽症
                    on_scene_time = np.random.lognormal(np.log(10.0), 0.5)
        else:
            # フォールバック: 傷病度別の標準時間
            if severity in ['重篤', '重症']:
                on_scene_time = np.random.lognormal(np.log(20.0), 0.5)
            elif severity == '中等症':
                on_scene_time = np.random.lognormal(np.log(15.0), 0.5)
            else:  # 軽症
                on_scene_time = np.random.lognormal(np.log(10.0), 0.5)
        
        # 3. 現場出発時刻
        depart_scene_time = arrive_scene_time + on_scene_time
        
        # 4. 病院選択と搬送時間
        hospital_h3 = self._select_hospital(call['h3_index'], severity)
        transport_time = self._calculate_travel_time(call['h3_index'], hospital_h3) / 60.0
        
        # 5. 病院到着時刻
        arrive_hospital_time = depart_scene_time + transport_time
        
        # 6. 病院滞在時間（ServiceTimeGeneratorを使用、call_datetimeも渡す）
        if self.service_time_generator:
            try:
                # ServiceTimeGeneratorEnhancedの場合はcall_datetimeを渡す
                import inspect
                sig = inspect.signature(self.service_time_generator.generate_time)
                if 'call_datetime' in sig.parameters:
                    # 拡張版の場合
                    hospital_time = self.service_time_generator.generate_time(
                        severity, 'hospital_time', call_datetime=call_datetime
                    )
                else:
                    # 従来版の場合
                    hospital_time = self.service_time_generator.generate_time(severity, 'hospital_time')
            except Exception as e:
                print(f"🚨 FALLBACK使用: 病院滞在時間生成エラー({severity}, hospital_time): {e}")
                print(f"   正確なサービス時間ではなく推定値を使用しています！")
                # フォールバック: ランダムな病院滞在時間
                if severity in ['重篤', '重症']:
                    hospital_time = np.random.lognormal(np.log(30.0), 0.5)
                elif severity == '中等症':
                    hospital_time = np.random.lognormal(np.log(20.0), 0.5)
                else:  # 軽症
                    hospital_time = np.random.lognormal(np.log(15.0), 0.5)
        else:
            # フォールバック: 傷病度別の標準時間
            if severity in ['重篤', '重症']:
                hospital_time = np.random.lognormal(np.log(30.0), 0.5)
            elif severity == '中等症':
                hospital_time = np.random.lognormal(np.log(20.0), 0.5)
            else:  # 軽症
                hospital_time = np.random.lognormal(np.log(15.0), 0.5)
        
        # 7. 病院出発時刻
        depart_hospital_time = arrive_hospital_time + hospital_time
        
        # 8. 帰署時間
        amb_state = self.ambulance_states[ambulance_id]
        return_time = self._calculate_travel_time(hospital_h3, amb_state['station_h3']) / 60.0
        
        # 9. 最終完了時刻
        completion_time = depart_hospital_time + return_time
        
        # 活動時間は実時間に合わせる（制限なし）
        
        if self.verbose_logging:
            print(f"救急車{ambulance_id}活動時間計算:")
            print(f"  応答: {response_time/60:.1f}分, 現場: {on_scene_time:.1f}分")
            print(f"  搬送: {transport_time:.1f}分, 病院: {hospital_time:.1f}分, 帰署: {return_time:.1f}分")
            print(f"  総活動時間: {completion_time - current_time:.1f}分")
        
        return completion_time
    
    def _select_hospital(self, scene_h3: str, severity: str) -> str:
        """傷病度に応じた病院選択（ValidationSimulatorと同じロジック）"""
        severe_conditions = ['重症', '重篤']
        
        # 重症・重篤の案件は決定論的選択
        if severity in severe_conditions:
            return self._select_hospital_deterministic(scene_h3, severity)
        
        # 軽症・中等症・死亡：確率的選択
        if not self.use_probabilistic_selection:
            return self._select_hospital_deterministic(scene_h3, severity)
        
        # 現在の時間情報を取得（エピソード内の時刻から算出）
        if hasattr(self, 'current_time_seconds'):
            current_time_seconds = self.current_time_seconds
        else:
            # フォールバックとして現在のステップから推定
            current_time_seconds = self.episode_step * 60.0
        
        current_hour = int((current_time_seconds / 3600) % 24)
        time_slot = current_hour // 4
        days_elapsed = int(current_time_seconds / 86400)
        day_of_week = days_elapsed % 7
        day_type = 'weekend' if day_of_week >= 5 else 'weekday'
        key = (time_slot, day_type, severity, scene_h3)
        
        # 1. 実績ベースの事前計算モデルから検索
        hospital_probs = self.hospital_selection_model.get(key) if hasattr(self, 'hospital_selection_model') else None
        
        if hospital_probs:
            pass  # 実績モデル使用
        else:
            # 2. 静的フォールバックモデルから検索
            if hasattr(self, 'static_fallback_model') and self.static_fallback_model:
                hospital_probs = self.static_fallback_model.get(severity, {}).get(scene_h3)
                if not hospital_probs:
                    # 静的フォールバックにもない場合は決定論的選択
                    return self._select_hospital_deterministic(scene_h3, severity)
            else:
                # フォールバックモデルもない場合
                return self._select_hospital_deterministic(scene_h3, severity)
        
        # 確率的選択の実行
        selected_hospital = self._probabilistic_selection(hospital_probs)
        
        if selected_hospital:
            return selected_hospital
        
        # 選択に失敗した場合は決定論的選択
        return self._select_hospital_deterministic(scene_h3, severity)
    
    def _select_hospital_deterministic(self, incident_h3: str, severity: str) -> str:
        """決定論的な病院選択（ValidationSimulatorと同じロジック）"""
        severe_conditions = ['重症', '重篤']
        
        if severity in severe_conditions:
            # 重症・重篤: 最寄りではなく、より適切な3次救急を選択
            if self.tertiary_hospitals:
                # 距離15km以内の3次救急から選択（最寄りではなくランダム選択）
                candidates = []
                inc_lat, inc_lon = h3.cell_to_latlng(incident_h3)
                
                for hospital_h3 in self.tertiary_hospitals:
                    try:
                        hosp_lat, hosp_lon = h3.cell_to_latlng(hospital_h3)
                        distance = self._calculate_distance(inc_lat, inc_lon, hosp_lat, hosp_lon)
                        if distance <= 15.0:  # 15km以内
                            candidates.append((hospital_h3, distance))
                    except:
                        continue
                
                if candidates:
                    # 上位3候補からランダム選択（実績の多様性を反映）
                    candidates.sort(key=lambda x: x[1])
                    top_candidates = candidates[:3]
                    selected = random.choice(top_candidates)[0]
                    return selected
        
        # 軽症・中等症・死亡の場合、または3次救急が見つからない重症・重篤ケース：2次以下から探す
        if self.secondary_primary_hospitals:
            nearest_secondary = self._find_nearest_hospital(incident_h3, self.secondary_primary_hospitals)
            if nearest_secondary:
                return nearest_secondary
        
        # それでも見つからない場合：軽症・中等症・死亡なら3次から探す
        if severity not in severe_conditions and self.tertiary_hospitals:
            nearest_tertiary = self._find_nearest_hospital(incident_h3, self.tertiary_hospitals)
            if nearest_tertiary:
                return nearest_tertiary
        
        # 全ての候補を探しても見つからない場合：現場と同じH3を返す
        return incident_h3
    
    def _find_nearest_hospital(self, incident_h3: str, hospital_candidates: set) -> Optional[str]:
        """指定された病院候補群から最寄りの病院を検索"""
        if not hospital_candidates:
            return None
        
        min_time = float('inf')
        nearest_hospital = None
        
        for hospital_h3 in hospital_candidates:
            try:
                travel_time = self._calculate_travel_time(incident_h3, hospital_h3)
                if travel_time < min_time:
                    min_time = travel_time
                    nearest_hospital = hospital_h3
            except:
                continue
        
        return nearest_hospital
    
    def _probabilistic_selection(self, hospital_probs: Dict[str, float]) -> Optional[str]:
        """確率分布に基づいて病院を選択"""
        if not hospital_probs:
            return None
        
        # NumPyの確率的選択を使用
        hospitals = list(hospital_probs.keys())
        probabilities = list(hospital_probs.values())
        
        # 確率値の型を修正（文字列が混入している場合の対処）
        try:
            probabilities = [float(p) for p in probabilities]
        except (ValueError, TypeError) as e:
            return None
        
        # 正規化（念のため）
        prob_sum = sum(probabilities)
        if prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]
        else:
            # 全て同じ確率
            probabilities = [1.0 / len(hospitals)] * len(hospitals)
        
        # 確率的選択
        selected_hospital = np.random.choice(hospitals, p=probabilities)
        
        return selected_hospital
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """2点間の距離をhaversine公式で計算（km単位）"""
        R = 6371  # 地球の半径（km）
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return distance
    
    # _calculate_travel_timeメソッドの修正
    def _calculate_travel_time(self, from_h3: str, to_h3: str) -> float:
        """
        移動時間を計算（秒単位）
        ValidationSimulatorのget_travel_timeと同じロジックを使用
        """
        # phaseは'response'をデフォルトとする（救急車選択時）
        phase = 'response'
        
        from_idx = self.grid_mapping.get(from_h3)
        to_idx = self.grid_mapping.get(to_h3)
        
        if from_idx is None or to_idx is None:
            # グリッドマッピングにない場合のフォールバック
            return 600.0  # デフォルト10分
        
        # 移動時間行列から取得
        current_travel_time_matrix = self.travel_time_matrices.get(phase)
        
        if current_travel_time_matrix is None:
            # responseフェーズの行列がない場合
            return 600.0  # デフォルト10分
        
        try:
            travel_time = current_travel_time_matrix[from_idx, to_idx]
            
            # 異常値チェック（ValidationSimulatorにはないが、安全のため）
            if travel_time <= 0 or travel_time > 3600:  # 1時間以上は異常
                return 600.0  # デフォルト10分
            
            return travel_time
        except:
            return 600.0  # エラー時のデフォルト
    

    
    def _calculate_reward(self, dispatch_result: Dict) -> float:
        """報酬を計算（RewardDesignerに完全委譲）"""
        if not dispatch_result['success']:
            # 失敗の種類に応じてペナルティを取得
            if dispatch_result.get('reason') == 'no_pending_call':
                return 0.0  # 事案なしは報酬なし
            elif dispatch_result.get('reason') == 'ambulance_busy':
                return self.reward_designer.get_failure_penalty('no_available')
            else:
                return self.reward_designer.get_failure_penalty('dispatch')
        
        # 成功時の報酬計算
        severity = dispatch_result['severity']
        response_time = dispatch_result['response_time']  # 秒単位
        
        # カバレッジ損失（L6, L13）を取得
        # state_encoderから取得、または直接計算
        L6 = dispatch_result.get('coverage_loss_6min', 0.0)
        L13 = dispatch_result.get('coverage_loss_13min', 0.0)
        
        # L6, L13がない場合は計算
        if L6 == 0.0 and L13 == 0.0 and hasattr(self, 'state_encoder'):
            # CompactStateEncoderを使用している場合
            if hasattr(self.state_encoder, 'get_selected_ambulance_coverage_loss'):
                action = dispatch_result.get('action', 0)
                L6, L13 = self.state_encoder.get_selected_ambulance_coverage_loss(
                    self.ambulance_states,
                    self.pending_call,
                    action
                )
        
        # RewardDesignerで報酬計算（新しいインターフェース）
        reward = self.reward_designer.calculate_step_reward(
            severity=severity,
            response_time_sec=response_time,
            L6=L6,
            L13=L13
        )
        
        # カバレッジ損失をdispatch_resultに追加（統計記録用）
        dispatch_result['coverage_loss_6min'] = L6
        dispatch_result['coverage_loss_13min'] = L13
        
        # デバッグ用ログ（最初の数回のみ）
        if hasattr(self, '_debug_reward_count'):
            self._debug_reward_count += 1
        else:
            self._debug_reward_count = 1
            
        if self._debug_reward_count <= 5:
            print(f"[報酬デバッグ] 傷病度: {severity}, 応答時間: {response_time/60:.1f}分, 報酬: {reward:.2f}")
            print(f"  - L6: {L6:.3f}, L13: {L13:.3f}")
        
        return reward
    
    def _update_statistics(self, dispatch_result: Dict):
        """統計情報を更新（拡張版）"""
        if not dispatch_result['success']:
            self.episode_stats['failed_dispatches'] += 1
            return
        
        self.episode_stats['total_dispatches'] += 1
        
        # 基本的な応答時間統計
        rt_minutes = dispatch_result['response_time_minutes']
        self.episode_stats['response_times'].append(rt_minutes)
        
        # 傷病度別統計
        severity = dispatch_result['severity']
        if severity not in self.episode_stats['response_times_by_severity']:
            self.episode_stats['response_times_by_severity'][severity] = []
        self.episode_stats['response_times_by_severity'][severity].append(rt_minutes)
        
        # 閾値達成率
        if rt_minutes <= 6.0:
            self.episode_stats['achieved_6min'] += 1
        if rt_minutes <= 13.0:
            self.episode_stats['achieved_13min'] += 1
        
        # 重症系の6分達成率
        if is_severe_condition(severity):
            self.episode_stats['critical_total'] += 1
            if rt_minutes <= 6.0:
                self.episode_stats['critical_6min'] += 1
        
        # 拡張統計の更新
        self._update_extended_statistics(dispatch_result)
        
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
    
    def _update_extended_statistics(self, dispatch_result: Dict):
        """拡張統計情報の更新"""
        try:
            ambulance_id = dispatch_result['ambulance_id']
            severity = dispatch_result['severity']
            rt_minutes = dispatch_result['response_time_minutes']
            
            # 救急車稼働統計
            if ambulance_id not in self.episode_stats['ambulance_utilization']['total_dispatches_by_ambulance']:
                self.episode_stats['ambulance_utilization']['total_dispatches_by_ambulance'][ambulance_id] = 0
            self.episode_stats['ambulance_utilization']['total_dispatches_by_ambulance'][ambulance_id] += 1
            
            # 時間別統計
            if self.pending_call and 'datetime' in self.pending_call:
                hour = self.pending_call['datetime'].hour
                self.episode_stats['temporal_patterns']['hourly_call_counts'][hour] += 1
                self.episode_stats['temporal_patterns']['hourly_response_times'][hour].append(rt_minutes)
                self.episode_stats['ambulance_utilization']['hourly_counts'][hour] += 1
            
            # 空間統計
            if self.pending_call and 'h3_index' in self.pending_call:
                h3_area = self.pending_call['h3_index']
                self.episode_stats['spatial_coverage']['areas_served'].add(h3_area)
                
                if h3_area not in self.episode_stats['spatial_coverage']['response_time_by_area']:
                    self.episode_stats['spatial_coverage']['response_time_by_area'][h3_area] = []
                    self.episode_stats['spatial_coverage']['call_density_by_area'][h3_area] = 0
                
                self.episode_stats['spatial_coverage']['response_time_by_area'][h3_area].append(rt_minutes)
                self.episode_stats['spatial_coverage']['call_density_by_area'][h3_area] += 1
            
            # 傷病度別詳細統計
            severity_category = self._get_severity_category(severity)
            if severity_category in self.episode_stats['severity_detailed_stats']:
                stats = self.episode_stats['severity_detailed_stats'][severity_category]
                stats['count'] += 1
                stats['response_times'].append(rt_minutes)
                if rt_minutes <= 6.0:
                    stats['under_6min'] += 1
                if rt_minutes <= 13.0:
                    stats['under_13min'] += 1
            
            # 移動距離の推定（簡易版）
            if hasattr(self, 'ambulance_states') and ambulance_id in self.ambulance_states:
                amb_state = self.ambulance_states[ambulance_id]
                if self.pending_call and 'h3_index' in self.pending_call:
                    # 距離行列から移動距離を取得（可能な場合）
                    estimated_distance = self._estimate_travel_distance(
                        amb_state['current_h3'], 
                        self.pending_call['h3_index']
                    )
                    self.episode_stats['efficiency_metrics']['total_distance'] += estimated_distance
                    
        except Exception as e:
            # 統計更新エラーは致命的ではないため、警告のみ出力
            print(f"統計更新でエラー: {e}")
    
    def _get_severity_category(self, severity: str) -> str:
        """傷病度から標準カテゴリに変換"""
        if severity in ['重篤', '重症', '死亡']:
            return 'critical'
        elif severity in ['中等症']:
            return 'moderate'
        elif severity in ['軽症']:
            return 'mild'
        else:
            return 'mild'  # デフォルト
    
    def _estimate_travel_distance(self, from_h3: str, to_h3: str) -> float:
        """移動距離の推定（km）"""
        try:
            from_idx = self.grid_mapping.get(from_h3)
            to_idx = self.grid_mapping.get(to_h3)
            
            if from_idx is not None and to_idx is not None and hasattr(self, 'travel_distance_matrix'):
                distance = self.travel_distance_matrix[from_idx, to_idx]
                return distance / 1000.0  # メートルからキロメートルに変換
            else:
                # フォールバック: 移動時間から距離を推定（平均時速30km/h）
                travel_time_seconds = self._calculate_travel_time(from_h3, to_h3)
                travel_time_hours = travel_time_seconds / 3600.0
                return travel_time_hours * 30.0  # 30km/h
        except:
            return 5.0  # デフォルト5km
    
    def get_episode_statistics(self) -> Dict:
        """エピソード統計を取得（RewardDesignerと連携、ハイブリッドモード対応）"""
        stats = self.episode_stats.copy()
        
        # 集計値の計算
        if stats['response_times']:
            total_calls = len(stats['response_times'])
            stats['summary'] = {
                'total_calls': total_calls,
                'mean_response_time': np.mean(stats['response_times']),
                'median_response_time': np.median(stats['response_times']),
                '95th_percentile_response_time': np.percentile(stats['response_times'], 95),
                '6min_achievement_rate': stats['achieved_6min'] / total_calls,
                '13min_achievement_rate': stats['achieved_13min'] / total_calls,
            }
            
            # 重症系達成率
            if stats['critical_total'] > 0:
                stats['summary']['critical_6min_rate'] = stats['critical_6min'] / stats['critical_total']
            else:
                stats['summary']['critical_6min_rate'] = 0.0
        
        # 救急車稼働率の計算
        if stats['ambulance_utilization']['total_dispatches_by_ambulance']:
            dispatches = list(stats['ambulance_utilization']['total_dispatches_by_ambulance'].values())
            stats['ambulance_utilization']['mean'] = np.mean(dispatches)
            stats['ambulance_utilization']['max'] = np.max(dispatches)
            stats['ambulance_utilization']['std'] = np.std(dispatches)
        
        # エリアカバレッジ
        stats['spatial_coverage']['areas_served'] = len(stats['spatial_coverage']['areas_served'])
        
        # 効率性メトリクス
        if stats['total_dispatches'] > 0:
            stats['efficiency_metrics']['distance_per_call'] = (
                stats['efficiency_metrics']['total_distance'] / stats['total_dispatches']
            )
        
        # ハイブリッドモード統計の追加
        if self.hybrid_mode:
            stats['hybrid_stats'] = {
                'direct_dispatch_count': self.direct_dispatch_count,
                'ppo_dispatch_count': self.ppo_dispatch_count,
                'direct_ratio': self.direct_dispatch_count / max(1, self.direct_dispatch_count + self.ppo_dispatch_count)
            }
        
        # エピソード報酬を計算
        if self.reward_designer:
            stats['episode_reward'] = self.reward_designer.calculate_episode_reward(stats)
        
        return stats
    
    def _advance_to_next_call(self):
        """次の事案へ進む（対応不能事案処理付き）"""
        # 現在の事案が対応不能になっていないかチェック
        if self.pending_call is not None:
            call_id = self.pending_call['id']
            if call_id in self.call_start_times:
                wait_time = self.episode_step - self.call_start_times[call_id]
                max_wait_time = self._get_max_wait_time(self.pending_call['severity'])
                
                if wait_time >= max_wait_time:
                    # 対応不能事案として記録
                    self._handle_unresponsive_call(self.pending_call, wait_time)
        
        self.episode_step += 1
        
        if self.episode_step < len(self.current_episode_calls):
            self.pending_call = self.current_episode_calls[self.episode_step]
            self.call_start_times[self.pending_call['id']] = self.episode_step
            
            # 時間経過に伴う救急車状態の更新
            self._update_ambulance_availability()
        else:
            self.pending_call = None
    
    def _update_ambulance_availability(self):
        """救急車の利用可能性を更新（validation_simulation互換版）"""
        # 救急車の復帰処理（ValidationSimulatorと同じロジック）
        returned_count = 0
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_state['status'] == 'dispatched':
                if 'call_completion_time' in amb_state and amb_state['call_completion_time'] is not None:
                    # 完了時刻に達した場合の復帰処理
                    if self.episode_step >= amb_state['call_completion_time']:
                        amb_state['status'] = 'available'
                        amb_state['current_h3'] = amb_state['station_h3']
                        amb_state['current_severity'] = None
                        amb_state['call_completion_time'] = None
                        returned_count += 1
                        if self.verbose_logging:
                            print(f"救急車{amb_id}が帰署完了 (ステップ{self.episode_step})")
                elif amb_state['last_dispatch_time'] is not None:
                    # フォールバック: 従来の方法（エラー防止）
                    elapsed = self.episode_step - amb_state['last_dispatch_time']
                    if elapsed >= 120:  # 最大2時間で強制復帰
                        amb_state['status'] = 'available'
                        amb_state['current_h3'] = amb_state['station_h3']
                        amb_state['current_severity'] = None
                        returned_count += 1
                        print(f"警告: 救急車{amb_id}を強制復帰 (2時間経過)")
        
        # 復帰した救急車がある場合はログ出力
        if returned_count > 0 and self.episode_step % 10 == 0:
            available_count = sum(1 for amb in self.ambulance_states.values() if amb['status'] == 'available')
            print(f"  {returned_count}台の救急車が復帰 (利用可能: {available_count}台)")
    
    def _get_max_wait_time(self, severity: str) -> int:
        """傷病度に応じた最大待機時間（分）- 現実的な救急システム"""
        if severity in ['重篤', '重症']:
            return 10  # 重症は10分で他地域から緊急応援
        elif severity == '中等症':
            return 20  # 中等症は20分で他地域応援
        else:  # 軽症
            return 45  # 軽症は45分で他地域応援（または搬送見送り）
    
    def _handle_unresponsive_call(self, call: Dict, wait_time: int):
        """対応不能事案の処理 - 現実的な救急システム"""
        severity = call['severity']
        
        # 重症度別の対応決定
        if severity in ['重篤', '重症']:
            response_type = 'emergency_support'  # 緊急応援（高速応答）
            support_time = 15 + wait_time  # 応援隊の到着時間（分）
            print(f"🚨 重症緊急応援: {severity} ({wait_time}分待機) → 他地域緊急隊が{support_time}分で対応")
        elif severity == '中等症':
            response_type = 'standard_support'  # 標準応援
            support_time = 25 + wait_time
            print(f"⚡ 中等症応援: {severity} ({wait_time}分待機) → 他地域隊が{support_time}分で対応")
        else:  # 軽症
            # 軽症は状況に応じて対応を分岐
            if wait_time > 60:
                response_type = 'transport_cancel'  # 搬送見送り
                support_time = None
                print(f"📋 軽症搬送見送り: {severity} ({wait_time}分待機) → 患者自力搬送または待機")
            else:
                response_type = 'delayed_support'  # 遅延応援
                support_time = 40 + wait_time
                print(f"🕐 軽症遅延応援: {severity} ({wait_time}分待機) → 他地域隊が{support_time}分で対応")
        
        # 対応不能事案として記録
        unhandled_call = {
            'call_id': call['id'],
            'severity': call['severity'],
            'wait_time': wait_time,
            'location': call.get('location', None),
            'handled_by': response_type,
            'support_time': support_time,
            'total_time': support_time if support_time else wait_time
        }
        self.unhandled_calls.append(unhandled_call)
        
        # 重症度別統計の更新
        self._update_unhandled_statistics(unhandled_call)
        
        # 重症度別ペナルティ（RewardDesignerに委譲）
        if self.reward_designer:
            penalty = self.reward_designer.calculate_unhandled_penalty(call['severity'], wait_time, response_type)
            if not hasattr(self, 'unhandled_penalty_total'):
                self.unhandled_penalty_total = 0
            self.unhandled_penalty_total += penalty
    
    def _update_unhandled_statistics(self, unhandled_call: Dict):
        """対応不能事案の詳細統計更新"""
        severity = unhandled_call['severity']
        response_type = unhandled_call['handled_by']
        
        # 重症度別統計
        if severity in ['重篤', '重症']:
            self.episode_stats['critical_unhandled'] = getattr(self.episode_stats, 'critical_unhandled', 0) + 1
            if response_type == 'emergency_support':
                self.episode_stats['critical_emergency_support'] = getattr(self.episode_stats, 'critical_emergency_support', 0) + 1
        elif severity == '中等症':
            self.episode_stats['moderate_unhandled'] = getattr(self.episode_stats, 'moderate_unhandled', 0) + 1
            if response_type == 'standard_support':
                self.episode_stats['moderate_standard_support'] = getattr(self.episode_stats, 'moderate_standard_support', 0) + 1
        else:  # 軽症
            self.episode_stats['mild_unhandled'] = getattr(self.episode_stats, 'mild_unhandled', 0) + 1
            if response_type == 'transport_cancel':
                self.episode_stats['mild_transport_cancel'] = getattr(self.episode_stats, 'mild_transport_cancel', 0) + 1
            elif response_type == 'delayed_support':
                self.episode_stats['mild_delayed_support'] = getattr(self.episode_stats, 'mild_delayed_support', 0) + 1
        
        # 全体統計
        self.episode_stats['unhandled_calls'] = getattr(self.episode_stats, 'unhandled_calls', 0) + 1
        self.episode_stats['total_support_time'] = getattr(self.episode_stats, 'total_support_time', 0) + unhandled_call.get('total_time', 0)
    
    def _calculate_coverage_impact(self, ambulance_id: Optional[int]) -> float:
        """
        カバレッジへの影響を簡易計算
        
        Returns:
            0.0-1.0の範囲（0=影響なし、1=大きな影響）
        """
        if ambulance_id is None:
            return 0.0
        
        # 利用可能な救急車の割合から簡易計算
        available_count = sum(1 for amb in self.ambulance_states.values() 
                             if amb['status'] == 'available')
        total_count = len(self.ambulance_states)
        
        if total_count == 0:
            return 0.0
        
        utilization_rate = 1.0 - (available_count / total_count)
        
        # 稼働率が高いほど、1台の出動の影響が大きい
        if utilization_rate > 0.8:
            return 0.8
        elif utilization_rate > 0.6:
            return 0.5
        elif utilization_rate > 0.4:
            return 0.3
        else:
            return 0.1
    

    
    def _is_episode_done(self) -> bool:
        """エピソード終了判定"""
        # 全事案を処理したら終了
        if self.pending_call is None:
            return True
        
        # ステップ数が最大値を超えたら終了
        if self.episode_step >= len(self.current_episode_calls):
            return True
        
        # 設定された最大ステップ数を超えたら終了（オプション）
        # configの階層構造に対応（data.max_steps_per_episode または max_steps_per_episode）
        max_steps = self.config.get('data', {}).get('max_steps_per_episode') or \
                    self.config.get('max_steps_per_episode') or \
                    3000  # フォールバック値
        if self.episode_step >= max_steps:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """現在の観測を取得"""
        state_dict = {
            'ambulances': self.ambulance_states,
            'pending_call': self.pending_call,
            'episode_step': self.episode_step,
            'time_of_day': self._get_time_of_day()
        }
        
        # ========== コンパクトモード: Top-K IDを更新 ==========
        if self.compact_mode:
            self.current_top_k_ids = self.state_encoder.get_top_k_ambulance_ids(
                state_dict['ambulances'],
                state_dict.get('pending_call')
            )
        
        # 初期化時に作成したインスタンスをそのまま使用する
        observation = self.state_encoder.encode_state(state_dict)
        
        return observation
    
    def _get_time_of_day(self) -> int:
        """現在の時刻を取得（0-23）"""
        if self.pending_call and 'datetime' in self.pending_call:
            return self.pending_call['datetime'].hour
        return 12  # デフォルト
    
    def _init_episode_stats(self) -> Dict:
        """エピソード統計の初期化（拡張版）"""
        return {
            # 基本統計
            'total_dispatches': 0,
            'failed_dispatches': 0,
            'response_times': [],
            'response_times_by_severity': {},
            'achieved_6min': 0,
            'achieved_13min': 0,
            'critical_total': 0,
            'critical_6min': 0,
            
            # 対応不能事案統計（詳細版）
            'unhandled_calls': 0,
            'critical_unhandled': 0,
            'moderate_unhandled': 0,
            'mild_unhandled': 0,
            'unhandled_penalty_total': 0.0,
            
            # 他地域応援統計
            'critical_emergency_support': 0,    # 重症緊急応援
            'moderate_standard_support': 0,     # 中等症標準応援
            'mild_delayed_support': 0,          # 軽症遅延応援
            'mild_transport_cancel': 0,         # 軽症搬送見送り
            'total_support_time': 0,            # 総応援対応時間
            
            # 救急車稼働統計
            'ambulance_utilization': {
                'hourly_counts': [0] * 24,  # 時間別出動回数
                'total_dispatches_by_ambulance': {},  # 救急車別出動回数
                'busy_time_by_ambulance': {},  # 救急車別稼働時間
            },
            
            # 空間統計
            'spatial_coverage': {
                'areas_served': set(),  # サービス提供エリア
                'response_time_by_area': {},  # エリア別応答時間
                'call_density_by_area': {},  # エリア別事案密度
            },
            
            # 時間パターン
            'temporal_patterns': {
                'hourly_call_counts': [0] * 24,  # 時間別事案数
                'hourly_response_times': {i: [] for i in range(24)},  # 時間別応答時間
            },
            
            # 効率性メトリクス
            'efficiency_metrics': {
                'total_distance': 0.0,  # 総移動距離
                'distance_per_call': 0.0,  # 事案あたり移動距離
                'travel_time_accuracy': [],  # 移動時間予測精度
            },
            
            # 傷病度別詳細統計
            'severity_detailed_stats': {
                'critical': {'count': 0, 'under_6min': 0, 'under_13min': 0, 'response_times': []},
                'moderate': {'count': 0, 'under_6min': 0, 'under_13min': 0, 'response_times': []},
                'mild': {'count': 0, 'under_6min': 0, 'under_13min': 0, 'response_times': []},
            }
        }
    
    def get_action_mask(self) -> np.ndarray:
        """利用可能な行動のマスクを取得（ハイブリッドv2対応版、コンパクトモード対応）"""
        
        # ========== コンパクトモード ==========
        if self.compact_mode:
            # Top-K用のマスク（基本的に全てTrue）
            mask = np.ones(self.action_dim, dtype=bool)
            
            # Top-Kに満たない場合は残りを無効化
            if self.current_top_k_ids:
                valid_count = len(self.current_top_k_ids)
                if valid_count < self.action_dim:
                    mask[valid_count:] = False
            
            return mask
        
        # ========== 従来モード ==========
        mask = np.zeros(self.action_dim, dtype=bool)
        
        # 基本マスク：利用可能な救急車
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_id < self.action_dim and amb_state['status'] == 'available':
                mask[amb_id] = True
        
        if self.pending_call is None:
            return mask
        
        severity = self.pending_call.get('severity', '')
        
        # ★★★ ハイブリッドモードv2対応 ★★★
        if self._is_hybrid_v2_enabled():
            # 重症系は全救急車を許可（直近隊選択はstep()で強制）
            if is_severe_condition(severity):
                return mask
            
            # 軽症系のフィルタリング（傷病度考慮運用と同等の条件）
            return self._get_filtered_mask_for_mild(mask, severity)
        
        # coverage_awareモードでアクションマスクが有効な場合、追加フィルタリング
        if (self.reward_designer.mode == 'coverage_aware'):
            
            action_mask_config = self.reward_designer.config.get('reward', {}).get('core', {}).get('action_mask', {})
            if action_mask_config.get('enabled', False):
                request_h3 = self.pending_call.get('h3_index')
                
                # 重症系の場合は全て許可（時間制約なし）
                if is_severe_condition(severity):
                    return mask
                
                # 軽症系の場合、時間制約とカバレッジ損失でフィルタ
                filtered_mask = np.zeros(self.action_dim, dtype=bool)
                available_ambulances = [amb_id for amb_id in range(self.action_dim) if mask[amb_id]]
                
                # 時間制約のチェック
                time_limit_seconds = 780  # 13分 = 780秒
                if action_mask_config.get('mild_time_limit_mask', False):
                    time_limit_seconds = self.reward_designer.coverage_aware_params.get('mild', {}).get('time_limit_seconds', 780)
                
                # カバレッジ損失閾値
                coverage_threshold = action_mask_config.get('coverage_loss_threshold', 0.8)
                use_coverage_mask = action_mask_config.get('coverage_loss_mask', False)
                
                for amb_id in available_ambulances:
                    amb_state = self.ambulance_states.get(amb_id)
                    if not amb_state:
                        continue
                    
                    # 応答時間をチェック
                    response_time = self._calculate_travel_time(
                        amb_state['current_h3'],
                        request_h3
                    )
                    
                    # 時間制約チェック
                    if response_time > time_limit_seconds:
                        continue
                    
                    # カバレッジ損失チェック
                    if use_coverage_mask:
                        try:
                            coverage_loss = self._calculate_coverage_loss(
                                amb_id,
                                available_ambulances,
                                request_h3
                            )
                            if coverage_loss >= coverage_threshold:
                                continue
                        except Exception:
                            # エラー時は許可（フォールバック）
                            pass
                    
                    filtered_mask[amb_id] = True
                
                # フィルタリング後も選択肢があるか確認
                if filtered_mask.sum() > 0:
                    return filtered_mask
                # 全てマスクされた場合は元のマスクを返す（最低限の選択肢を確保）
        
        return mask
 
    def get_best_action_for_call(self) -> Optional[int]:
        """
        現在の事案に対して最適な救急車（行動）を選択
        学習初期はこれを教師として使用できる
        """
        if self.pending_call is None:
            return None
        
        best_action = None
        min_travel_time = float('inf')
        
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_state['status'] != 'available':
                continue
            
            # 移動時間を計算
            travel_time = self._calculate_travel_time(
                amb_state['current_h3'],
                self.pending_call['h3_index']
            )
            
            if travel_time < min_travel_time:
                min_travel_time = travel_time
                best_action = amb_id
        
        return best_action
    
    def render(self, mode: str = 'human'):
        """環境の可視化（オプション）"""
        if mode == 'human':
            print(f"\nStep {self.episode_step}")
            if self.pending_call:
                print(f"  事案: {self.pending_call['severity']} at {self.pending_call['h3_index']}")
            
            available_count = sum(1 for a in self.ambulance_states.values() if a['status'] == 'available')
            print(f"  利用可能救急車: {available_count}/{len(self.ambulance_states)}")
            
            if self.episode_stats['total_dispatches'] > 0:
                avg_rt = np.mean(self.episode_stats['response_times'])
                rate_6min = self.episode_stats['achieved_6min'] / self.episode_stats['total_dispatches'] * 100
                print(f"  平均応答時間: {avg_rt:.1f}分")
                print(f"  6分達成率: {rate_6min:.1f}%")
    
    def _create_virtual_ambulances_if_needed(self, actual_ambulances: pd.DataFrame) -> pd.DataFrame:
        """
        必要に応じて仮想救急車を作成
        
        Args:
            actual_ambulances: 実際の救急車データ
            
        Returns:
            仮想救急車を含む救急車データ
        """
        # 設定から仮想救急車パラメータを取得
        data_config = self.config.get('data', {})
        virtual_count = data_config.get('virtual_ambulances', None)
        multiplier = data_config.get('ambulance_multiplier', 1.0)
        
        if virtual_count and virtual_count > 0:
            # 仮想救急車を追加（既存の救急車は保持）
            target_count = len(actual_ambulances) + virtual_count
            print(f"  仮想救急車追加: {len(actual_ambulances)}台 → {target_count}台 (追加: {virtual_count}台)")
            return self._create_virtual_ambulances(actual_ambulances, target_count)
        elif multiplier > 1.0:
            # 既存の救急車を複製
            target_count = int(len(actual_ambulances) * multiplier)
            print(f"  救急車複製: {len(actual_ambulances)}台 → {target_count}台 (倍率: {multiplier})")
            return self._duplicate_ambulances(actual_ambulances, target_count)
        else:
            # 仮想救急車は作成しない
            return actual_ambulances
    
    def _create_virtual_ambulances(self, actual_ambulances: pd.DataFrame, target_count: int) -> pd.DataFrame:
        """
        既存の救急署から均等に仮想救急車を追加
        
        Args:
            actual_ambulances: 実際の救急車データ
            target_count: 目標救急車数
            
        Returns:
            仮想救急車を含む救急車データ
        """
        result_ambulances = actual_ambulances.copy()
        
        # 救急署ごとにグループ化（H3インデックスで）
        stations = {}
        for _, amb in actual_ambulances.iterrows():
            try:
                # H3インデックスを計算
                lat = float(amb['latitude'])
                lng = float(amb['longitude'])
                station_h3 = h3.latlng_to_cell(lat, lng, 9)
                
                if station_h3 not in stations:
                    stations[station_h3] = []
                stations[station_h3].append(amb)
            except Exception as e:
                print(f"警告: 救急車のH3計算エラー: {e}")
                continue
        
        print(f"  救急署数: {len(stations)}署")
        print(f"  実際の救急車数: {len(actual_ambulances)}台")
        
        # 各署に仮想救急車を均等に追加
        virtual_id_counter = len(actual_ambulances)
        
        while len(result_ambulances) < target_count:
            for station_h3, station_ambs in stations.items():
                if len(result_ambulances) >= target_count:
                    break
                
                # この署に仮想救急車を1台追加
                base_ambulance = station_ambs[0]  # 代表的な救急車をベースにする
                virtual_ambulance = base_ambulance.copy()
                
                # 仮想救急車の識別情報を更新
                virtual_ambulance['id'] = f"virtual_{virtual_id_counter}"
                virtual_ambulance['name'] = f"virtual_team_{virtual_id_counter}"
                virtual_ambulance['team_name'] = f"virtual_team_{virtual_id_counter}"
                virtual_ambulance['is_virtual'] = True
                
                # 同じ署の位置を使用（位置は変更しない）
                # 必要に応じて微細な位置調整も可能
                lat_offset = np.random.uniform(-0.001, 0.001)  # 約100m以内
                lng_offset = np.random.uniform(-0.001, 0.001)  # 約100m以内
                
                virtual_ambulance['latitude'] = float(virtual_ambulance['latitude']) + lat_offset
                virtual_ambulance['longitude'] = float(virtual_ambulance['longitude']) + lng_offset
                
                # 座標の有効性チェック
                if (-90 <= virtual_ambulance['latitude'] <= 90 and 
                    -180 <= virtual_ambulance['longitude'] <= 180):
                    
                    # 仮想救急車を追加
                    result_ambulances = pd.concat([result_ambulances, virtual_ambulance.to_frame().T], 
                                                ignore_index=True)
                    virtual_id_counter += 1
                    
                    print(f"  仮想救急車{virtual_id_counter-1}を署{station_h3}に追加")
        
        return result_ambulances
    
    def _duplicate_ambulances(self, actual_ambulances: pd.DataFrame, target_count: int) -> pd.DataFrame:
        """
        既存の救急車を複製
        
        Args:
            actual_ambulances: 実際の救急車データ
            target_count: 目標救急車数
            
        Returns:
            複製された救急車データ
        """
        result_ambulances = actual_ambulances.copy()
        
        # 複製カウンタ
        duplicate_counter = 0
        
        while len(result_ambulances) < target_count:
            for _, base_ambulance in actual_ambulances.iterrows():
                if len(result_ambulances) >= target_count:
                    break
                
                # 既存の救急車を複製
                duplicate_ambulance = base_ambulance.copy()
                
                # 複製救急車の識別情報を更新
                duplicate_ambulance['id'] = f"duplicate_{duplicate_counter}"
                duplicate_ambulance['name'] = f"複製救急車{duplicate_counter}"
                
                # 位置を少しずらす（半径300m以内のランダムな位置）
                lat_offset = np.random.uniform(-0.0027, 0.0027)  # 約300m
                lng_offset = np.random.uniform(-0.0027, 0.0027)  # 約300m
                
                duplicate_ambulance['latitude'] = float(duplicate_ambulance['latitude']) + lat_offset
                duplicate_ambulance['longitude'] = float(duplicate_ambulance['longitude']) + lng_offset
                
                # 座標の有効性チェック
                if (-90 <= duplicate_ambulance['latitude'] <= 90 and 
                    -180 <= duplicate_ambulance['longitude'] <= 180):
                    
                    # 複製救急車を追加
                    result_ambulances = pd.concat([result_ambulances, duplicate_ambulance.to_frame().T], 
                                                ignore_index=True)
                    duplicate_counter += 1
                    
                    if len(result_ambulances) >= target_count:
                        break
        
        return result_ambulances