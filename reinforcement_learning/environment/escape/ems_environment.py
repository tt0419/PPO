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
from data_cache import get_emergency_data_cache
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 統一された傷病度定数をインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from constants import SEVERITY_GROUPS, is_severe_condition

from validation_simulation import (
    ValidationSimulator,
    EventType,
    AmbulanceStatus,
    EmergencyCall,
    Event,
    ServiceTimeGenerator
)

class HierarchicalServiceTimeGenerator:
    """階層的パラメータファイル対応のServiceTimeGenerator"""
    
    def __init__(self, params_file: str):
        import json
        with open(params_file, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
    
    def generate_time(self, severity: str, phase: str) -> float:
        """指定されたフェーズの時間を生成（分単位）"""
        
        # severityがパラメータに存在しない場合、'軽症'にフォールバック
        severity_params = self.params.get(severity, self.params.get('軽症', {}))
        
        # フェーズが存在しない場合のフォールバック
        if phase not in severity_params:
            default_times = {
                'on_scene_time': 15.0,
                'hospital_time': 20.0,
                'return_time': 10.0
            }
            return np.random.lognormal(np.log(default_times.get(phase, 10.0)), 0.5)
        
        phase_params = severity_params[phase]
        
        # 階層構造の場合は'default'キーを使用
        if isinstance(phase_params, dict) and 'default' in phase_params:
            default_params = phase_params['default']
            if default_params['distribution'] == 'lognormal':
                return np.random.lognormal(default_params['mu'], default_params['sigma'])
            else:
                return default_params.get('mean_minutes', 15.0)
        # 従来の単純構造の場合
        elif isinstance(phase_params, dict) and 'distribution' in phase_params:
            if phase_params['distribution'] == 'lognormal':
                return np.random.lognormal(phase_params['mu'], phase_params['sigma'])
            else:
                return phase_params.get('mean_minutes', 15.0)
        else:
            # 構造が不明な場合
            print(f"⚠️ 不明なパラメータ構造: {severity}.{phase} = {type(phase_params)}")
            default_times = {
                'on_scene_time': 15.0,
                'hospital_time': 20.0,
                'return_time': 10.0
            }
            return np.random.lognormal(np.log(default_times.get(phase, 10.0)), 0.5)

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
        
        # データパスの設定
        self.base_dir = Path("data/tokyo")
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
        
        # 状態・行動空間の次元
        self.action_dim = len(self.ambulance_data)  # 実際の救急車数
        
        # ★★★【修正提案】★★★
        # StateEncoderの初期化をここで行い、インスタンスをクラス変数として保持する
        response_matrix = self.travel_time_matrices.get('response', None)
        if response_matrix is None:
            print("警告: responseフェーズの移動時間行列が見つかりません。")

        # StateEncoderを初期化して、self.state_encoderとして保持
        from .state_encoder import StateEncoder
        self.state_encoder = StateEncoder(
            config=self.config,
            max_ambulances=self.action_dim,
            travel_time_matrix=response_matrix,
            grid_mapping=self.grid_mapping
        )
        
        # StateEncoderインスタンスから状態次元を取得する
        self.state_dim = self.state_encoder.state_dim
        # ★★★【修正ここまで】★★★
        
        print(f"状態空間次元: {self.state_dim}")
        print(f"行動空間次元: {self.action_dim}")
        
        # 統計情報の初期化
        self.episode_stats = self._init_episode_stats()
        
        # RewardDesignerを一度だけ初期化
        from .reward_designer import RewardDesigner
        self.reward_designer = RewardDesigner(self.config)
        
        # ServiceTimeGeneratorの初期化
        self._init_service_time_generator()        
        
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
        """ServiceTimeGeneratorの初期化"""
        # サービス時間パラメータファイルの検索
        possible_params_paths = [
            self.base_dir / "service_time_analysis/lognormal_parameters_hierarchical.json",
            self.base_dir / "service_time_analysis/lognormal_parameters.json",
            "data/tokyo/service_time_analysis/lognormal_parameters_hierarchical.json",
            "data/tokyo/service_time_analysis/lognormal_parameters.json"
        ]
        
        params_file = None
        for path in possible_params_paths:
            if Path(path).exists():
                params_file = str(path)
                print(f"  サービス時間パラメータ読み込み: {params_file}")
                break
        
        if params_file:
            try:
                # 階層的パラメータファイルの場合は専用クラスを使用
                if 'hierarchical' in params_file:
                    self.service_time_generator = HierarchicalServiceTimeGenerator(params_file)
                    print("  ✓ HierarchicalServiceTimeGenerator初期化成功")
                else:
                    self.service_time_generator = ServiceTimeGenerator(params_file)
                    print("  ✓ ServiceTimeGenerator初期化成功")
            except Exception as e:
                print(f"  ❌ ServiceTimeGenerator初期化失敗: {e}")
                print(f"  フォールバック処理を使用します")
                self.service_time_generator = None
        else:
            print("  ❌ サービス時間パラメータファイルが見つかりません")
            print("  フォールバック処理を使用します")
            self.service_time_generator = None
    
    def _load_base_data(self):
        """基本データの読み込み"""
        print("\n基本データ読み込み中...")
        
        # 救急署データ
        firestation_path = self.base_dir / "import/amb_place_master.csv"
        ambulance_data_full = pd.read_csv(firestation_path, encoding='utf-8')
        ambulance_data_full = ambulance_data_full[ambulance_data_full['special_flag'] == 1]
        
        # エリア制限フィルタリングの設定確認
        area_restriction = self.config.get('data', {}).get('area_restriction', {})
        if area_restriction.get('enabled', False):
            section_code = area_restriction.get('section_code')
            area_name = area_restriction.get('area_name', '指定エリア')
            
            # section_codeがnullまたはNoneの場合は全方面を使用（東京23区全域など）
            if section_code is None or section_code == 'null':
                print(f"  {area_name}（全方面）を使用")
                # 不要な救急隊を除外（救急隊なし、デイタイム）
                if 'team_name' in ambulance_data_full.columns:
                    before_team_filter = len(ambulance_data_full)
                    # '救急隊なし'と'デイタイム'を含む隊を除外
                    team_mask = (
                        (ambulance_data_full['team_name'] != '救急隊なし') &
                        (~ambulance_data_full['team_name'].str.contains('デイタイム', na=False))
                    )
                    self.ambulance_data = ambulance_data_full[team_mask].copy()
                    print(f"  チーム名フィルタ適用: {before_team_filter}台 → {len(self.ambulance_data)}台 (救急隊なし・デイタイム除外)")
                else:
                    self.ambulance_data = ambulance_data_full
            elif section_code in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                # 指定方面の救急隊に限定
                before_filter = len(ambulance_data_full)
                section_filtered = ambulance_data_full[ambulance_data_full['section'] == section_code].copy()
                
                # 不要な救急隊を除外（救急隊なし、デイタイム）
                if 'team_name' in section_filtered.columns:
                    before_team_filter = len(section_filtered)
                    # '救急隊なし'と'デイタイム'を含む隊を除外
                    team_mask = (
                        (section_filtered['team_name'] != '救急隊なし') &
                        (~section_filtered['team_name'].str.contains('デイタイム', na=False))
                    )
                    section_filtered = section_filtered[team_mask].copy()
                    print(f"  チーム名フィルタ適用: {before_team_filter}台 → {len(section_filtered)}台 (救急隊なし・デイタイム除外)")
                
                self.ambulance_data = section_filtered
                print(f"  {area_name}フィルタ適用: {before_filter}台 → {len(self.ambulance_data)}台")
                
                if len(self.ambulance_data) == 0:
                    print(f"  警告: {area_name}の救急車が見つかりません。全体を使用します。")
                    self.ambulance_data = ambulance_data_full
            else:
                # その他の場合は全体を使用
                self.ambulance_data = ambulance_data_full
        else:
            self.ambulance_data = ambulance_data_full
            
        print(f"  救急署数: {len(self.ambulance_data)}")
        
        # 病院データ（方面に関係なく全体を使用）
        hospital_path = self.base_dir / "import/hospital_master.csv"
        self.hospital_data = pd.read_csv(hospital_path, encoding='utf-8')
        print(f"  病院数: {len(self.hospital_data)}")
        
        # グリッドマッピング
        grid_mapping_path = self.base_dir / "processed/grid_mapping_res9.json"
        with open(grid_mapping_path, 'r', encoding='utf-8') as f:
            self.grid_mapping = json.load(f)
        print(f"  H3グリッド数: {len(self.grid_mapping)}")
        
        # 移動時間行列（軽量版 - 学習用）
        # 実際のファイルパスに合わせて修正してください
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
        self.max_steps_per_episode = len(self.current_episode_calls)
        
        print(f"読み込まれた事案数: {len(self.current_episode_calls)}")
        
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
        
        print(f"  救急車データから初期化開始: {len(self.ambulance_data)}台のデータ")
        
        # DataFrameのindexではなく、0から始まる連続した番号を使用
        for amb_id, (_, row) in enumerate(self.ambulance_data.iterrows()):
            if amb_id >= self.action_dim:
                break
            
            try:
                # 座標の検証
                lat = float(row['latitude'])
                lng = float(row['longitude'])
                
                if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                    print(f"    ⚠️ 救急車{amb_id}: 無効な座標 lat={lat}, lng={lng}")
                    continue
                
                h3_index = h3.latlng_to_cell(lat, lng, 9)
                
                self.ambulance_states[amb_id] = {
                    'id': f"amb_{amb_id}",
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
        
        # 初期化直後のマスクチェック
        initial_mask = self.get_action_mask()
        print(f"  初期化直後の利用可能数: {initial_mask.sum()}台")
    
    def step(self, action: int) -> StepResult:
        """
        環境のステップ実行
        
        Args:
            action: 選択された救急車のインデックス
            
        Returns:
            StepResult: 観測、報酬、終了フラグ、追加情報
        """
        try:
            # デバッグ用: 最適行動との比較を出力
            if hasattr(self, 'verbose_logging') and self.verbose_logging:
                optimal_action = self.get_optimal_action()
                if optimal_action is not None and action != optimal_action:
                    optimal_time = self._calculate_travel_time(
                        self.ambulance_states[optimal_action]['current_h3'],
                        self.pending_call['h3_index']
                    )
                    actual_time = self._calculate_travel_time(
                        self.ambulance_states[action]['current_h3'],
                        self.pending_call['h3_index']
                    )
                    print(f"[選択比較] PPO選択: 救急車{action}({actual_time/60:.1f}分) "
                        f"vs 最適: 救急車{optimal_action}({optimal_time/60:.1f}分)")
            
            # 行動の実行（救急車の配車）
            dispatch_result = self._dispatch_ambulance(action)
            
            # 報酬の計算
            reward = self._calculate_reward(dispatch_result)
            
            # 統計情報の更新
            self._update_statistics(dispatch_result)
            
            # 次の事案へ進む
            self._advance_to_next_call()
            
            # エピソード終了判定
            done = self._is_episode_done()
            
            # 次の観測を取得
            observation = self._get_observation()
            
            # 追加情報
            info = {
                'dispatch_result': dispatch_result,
                'episode_stats': self.episode_stats.copy(),
                'step': self.episode_step
            }
            
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
            最適な救急車のID、または None
        """
        if self.pending_call is None:
            return None
        
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


    
    def _dispatch_ambulance(self, action: int) -> Dict:
        """救急車を配車"""
        if self.pending_call is None:
            return {'success': False, 'reason': 'no_pending_call'}
        
        # 行動の妥当性チェック
        if action >= len(self.ambulance_states):
            return {'success': False, 'reason': 'invalid_action'}
        
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
            'matched_teacher': self.current_matched_teacher
        }
        
        return result
    
    def _calculate_ambulance_completion_time(self, ambulance_id: int, call: Dict, response_time: float) -> float:
        """救急車の活動完了時間を計算（ValidationSimulator互換）"""
        current_time = self.episode_step  # 現在時刻（分単位）
        severity = call['severity']
        
        # 1. 現場到着時刻 = 現在時刻 + 応答時間
        arrive_scene_time = current_time + (response_time / 60.0)
        
        # 2. 現場活動時間（ServiceTimeGeneratorを使用）
        if self.service_time_generator:
            try:
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
        
        # 6. 病院滞在時間（ServiceTimeGeneratorを使用）
        if self.service_time_generator:
            try:
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
        
        if self.verbose_logging:
            print(f"救急車{ambulance_id}活動時間計算:")
            print(f"  応答: {response_time/60:.1f}分, 現場: {on_scene_time:.1f}分")
            print(f"  搬送: {transport_time:.1f}分, 病院: {hospital_time:.1f}分, 帰署: {return_time:.1f}分")
            print(f"  総活動時間: {completion_time - current_time:.1f}分")
        
        return completion_time
    
    def _select_hospital(self, scene_h3: str, severity: str) -> str:
        """病院選択（ValidationSimulatorの簡易版）"""
        # 現在は最も近い病院を選択（実際のロジックはより複雑）
        if not hasattr(self, '_hospital_h3_list'):
            self._hospital_h3_list = []
            for _, hospital in self.hospital_data.iterrows():
                try:
                    if pd.notna(hospital['latitude']) and pd.notna(hospital['longitude']):
                        h3_idx = h3.latlng_to_cell(hospital['latitude'], hospital['longitude'], 9)
                        if h3_idx in self.grid_mapping:
                            self._hospital_h3_list.append(h3_idx)
                except:
                    continue
        
        if not self._hospital_h3_list:
            return scene_h3  # フォールバック
        
        # 最短距離の病院を選択
        min_distance = float('inf')
        best_hospital_h3 = self._hospital_h3_list[0]
        
        for hospital_h3 in self._hospital_h3_list:
            try:
                distance = self._calculate_travel_time(scene_h3, hospital_h3)
                if distance < min_distance:
                    min_distance = distance
                    best_hospital_h3 = hospital_h3
            except:
                continue
        
        return best_hospital_h3
    
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
        response_time = dispatch_result['response_time']
        
        # カバレッジ影響の計算（簡易版）
        coverage_impact = self._calculate_coverage_impact(dispatch_result.get('ambulance_id'))
        
        # 追加情報（教師との一致など）
        additional_info = {
            'matched_teacher': dispatch_result.get('matched_teacher', False),
            'distance_rank': dispatch_result.get('distance_rank', None)
        }
        
        # RewardDesignerで報酬計算
        reward = self.reward_designer.calculate_step_reward(
            severity=severity,
            response_time=response_time,
            coverage_impact=coverage_impact,
            additional_info=additional_info
        )
        
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
        """エピソード統計を取得（RewardDesignerと連携）"""
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
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_state['status'] == 'dispatched':
                if 'call_completion_time' in amb_state and amb_state['call_completion_time'] is not None:
                    # 完了時刻に達した場合の復帰処理
                    if self.episode_step >= amb_state['call_completion_time']:
                        amb_state['status'] = 'available'
                        amb_state['current_h3'] = amb_state['station_h3']
                        amb_state['current_severity'] = None
                        amb_state['call_completion_time'] = None
                        if self.verbose_logging:
                            print(f"救急車{amb_id}が帰署完了 (ステップ{self.episode_step})")
                elif amb_state['last_dispatch_time'] is not None:
                    # フォールバック: 従来の方法（エラー防止）
                    elapsed = self.episode_step - amb_state['last_dispatch_time']
                    if elapsed >= 120:  # 最大2時間で強制復帰
                        amb_state['status'] = 'available'
                        amb_state['current_h3'] = amb_state['station_h3']
                        amb_state['current_severity'] = None
                        print(f"警告: 救急車{amb_id}を強制復帰 (2時間経過)")
    
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
        max_steps = self.config.get('max_steps_per_episode', 1000)
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
        """利用可能な行動のマスクを取得"""
        mask = np.zeros(self.action_dim, dtype=bool)
        
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_id < self.action_dim and amb_state['status'] == 'available':
                mask[amb_id] = True
        
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