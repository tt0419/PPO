# validation_simulation.py

import os
import numpy as np
import pandas as pd
import heapq
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json
import h3
from tqdm import tqdm
# ★★★ 修正: matplotlibバックエンドを非インタラクティブに設定 ★★★
import matplotlib
matplotlib.use('Agg')  # ファイル出力専用バックエンド

import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import csv
from collections import defaultdict
import seaborn as sns
import sys

# 現在のプロジェクトディレクトリ（05_Ambulance_RL）を取得
# ファイル構造: 05_Ambulance_RL/validation_simulation.py
CURRENT_PROJECT_DIR = Path(__file__).resolve().parent
if str(CURRENT_PROJECT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_PROJECT_DIR))

# 後方互換性のため fix_dir も同じディレクトリを参照
fix_dir = CURRENT_PROJECT_DIR

# 親ディレクトリ（必要な場合のみ）
PROJECT_ROOT = CURRENT_PROJECT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_cache import get_emergency_data_cache, get_datetime_range_emergency_data

# ServiceTimeGeneratorEnhancedのインポート
# 現在のプロジェクトディレクトリ（05_Ambulance_RL）を取得
CURRENT_PROJECT_DIR = Path(__file__).resolve().parent  # 05_Ambulance_RL ディレクトリ
service_time_analysis_path = CURRENT_PROJECT_DIR / "data" / "tokyo" / "service_time_analysis"
if str(service_time_analysis_path) not in sys.path:
    sys.path.append(str(service_time_analysis_path))

try:
    from service_time_generator_enhanced import ServiceTimeGeneratorEnhanced
    USE_ENHANCED_GENERATOR = True
except ImportError:
    print("警告: ServiceTimeGeneratorEnhancedが見つかりません。従来版を使用します。")
    print(f"検索パス: {service_time_analysis_path}")
    USE_ENHANCED_GENERATOR = False

# ディスパッチ戦略のインポート
from dispatch_strategies import (
    DispatchStrategy, 
    StrategyFactory,
    EmergencyRequest,
    AmbulanceInfo,
    DispatchContext,
    DispatchPriority,
    PPOStrategy,  # ★★★ PPO戦略を追加 ★★★
    STRATEGY_CONFIGS
)

plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['font.size'] = 14  # デフォルトフォントサイズを大きく設定
plt.rcParams['axes.titlesize'] = 18  # タイトルのフォントサイズ
plt.rcParams['axes.labelsize'] = 16  # 軸ラベルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 14  # x軸目盛りのフォントサイズ
plt.rcParams['ytick.labelsize'] = 14  # y軸目盛りのフォントサイズ
plt.rcParams['legend.fontsize'] = 14  # 凡例のフォントサイズ

class EventType(Enum):
    NEW_CALL = "new_call"
    DISPATCH = "dispatch"
    ARRIVE_SCENE = "arrive_scene"
    DEPART_SCENE = "depart_scene"
    ARRIVE_HOSPITAL = "arrive_hospital"
    AMBULANCE_AVAILABLE = "ambulance_available"
    BREAK_END = "break_end"  # 休憩終了イベントを追加

class AmbulanceStatus(Enum):
    AVAILABLE = "available"
    DISPATCHED = "dispatched"
    ON_SCENE = "on_scene"
    TRANSPORTING = "transporting"
    AT_HOSPITAL = "at_hospital"
    RETURNING = "returning"
    ON_BREAK = "on_break"  # 休憩中ステータスを追加

@dataclass
class Event:
    time: float
    event_type: EventType
    data: Dict
    
    def __lt__(self, other):
        return self.time < other.time

@dataclass
class Ambulance:
    id: str
    team_name: str
    station_h3_index: str
    current_h3_index: str
    status: AmbulanceStatus = AmbulanceStatus.AVAILABLE
    assigned_call: Optional[str] = None
    total_active_time: float = 0.0
    total_distance: float = 0.0
    num_calls_handled: int = 0
    
    # 休憩関連フィールドを追加
    section: int = 1  # 方面番号（1-10）
    last_rest_time: Optional[float] = None  # 最後に休憩を取った時刻
    last_crew_change_time: Optional[float] = None  # 最後の乗務員交代時刻
    
    # レストタイム判定用
    lunch_period_standby_time: float = 0.0  # 11-13時の待機時間累計
    lunch_extended_standby_time: float = 0.0  # 11-15時の待機時間累計
    dinner_period_standby_time: float = 0.0  # 17-19時の待機時間累計
    dinner_extended_standby_time: float = 0.0  # 17-20時の待機時間累計
    
    # 深夜インターバル用
    night_activity_start_time: Optional[float] = None
    night_activity_duration: float = 0.0
    night_activity_count: int = 0
    
    # 休憩状態
    is_on_break: bool = False
    break_end_time: Optional[float] = None
    break_type: Optional[str] = None  # 'rest_time' or 'interval'

@dataclass
class EmergencyCall:
    id: str
    time: float  # 事案覚知時刻 (秒)
    h3_index: str
    severity: str
    assigned_ambulance: Optional[str] = None
    
    # 覚知年月日時分を追加
    call_datetime: Optional[pd.Timestamp] = None
    
    # 時刻記録フィールド
    dispatch_time: Optional[float] = None
    arrive_scene_time: Optional[float] = None
    depart_scene_time: Optional[float] = None
    arrive_hospital_time: Optional[float] = None
    depart_hospital_time: Optional[float] = None
    return_to_station_time: Optional[float] = None
    completion_time: Optional[float] = None

    # 算出した時間 (分単位)
    response_time: Optional[float] = None
    on_scene_duration: Optional[float] = None
    transport_duration: Optional[float] = None
    hospital_duration: Optional[float] = None
    return_duration: Optional[float] = None

    # 移動距離 (キロメートル単位)
    dispatch_to_scene_distance: Optional[float] = None
    scene_to_hospital_distance: Optional[float] = None
    hospital_to_station_distance: Optional[float] = None
    total_distance: Optional[float] = None

class ServiceTimeGenerator:
    """傷病度別サービス時間生成器"""
    
    def __init__(self, params_file: str):
        with open(params_file, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
    
    def generate_time(self, severity: str, phase: str) -> float:
        """指定されたフェーズの時間を生成（分単位）"""
        
        # severityがパラメータに存在しない場合、'その他'を試し、それもなければ'軽症'にフォールバック
        severity_params = self.params.get(severity, self.params.get('その他', self.params['軽症']))
        
        # さらにphaseがその傷病度パラメータにない場合のフォールバック（元のロジックを維持）
        if phase not in severity_params:
            default_times = {
                'on_scene_time': 15.0,
                'hospital_time': 20.0,
                'return_time': 10.0
            }
            return np.random.lognormal(np.log(default_times.get(phase, 10.0)), 0.5)
        
        phase_params = severity_params[phase]
        
        if phase_params['distribution'] == 'lognormal':
            return np.random.lognormal(phase_params['mu'], phase_params['sigma'])
        else:
            return phase_params.get('mean_minutes', 15.0)

class TravelTimeAnalyzer:
    """移動時間の詳細分析クラス"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.travel_time_logs = []
        self.same_grid_counts = defaultdict(int)
        self.distance_categories = {
            'same_grid': [],
            'adjacent_grid': [],
            'near_distance': [],
            'medium_distance': [],
            'far_distance': []
        }
        
    def log_travel_time(self, from_h3: str, to_h3: str, phase: str, 
                       travel_time_seconds: float, distance_km: float = None, 
                       context: str = 'unknown'):
        """移動時間の詳細ログを記録"""
        
        # 距離カテゴリの判定
        is_same_grid = (from_h3 == to_h3)
        is_adjacent = False
        distance_category = 'unknown'
        
        if is_same_grid:
            distance_category = 'same_grid'
            self.same_grid_counts[phase] += 1
        elif distance_km is not None:
            if distance_km <= 0.5:
                distance_category = 'adjacent_grid'
                # H3の隣接グリッド判定（簡易版）
                is_adjacent = self._check_adjacent_grids(from_h3, to_h3)
            elif distance_km <= 2.0:
                distance_category = 'near_distance'
            elif distance_km <= 5.0:
                distance_category = 'medium_distance'
            else:
                distance_category = 'far_distance'
        
        log_entry = {
            'from_h3': from_h3,
            'to_h3': to_h3,
            'phase': phase,
            'travel_time_seconds': travel_time_seconds,
            'travel_time_minutes': travel_time_seconds / 60.0,
            'distance_km': distance_km,
            'distance_category': distance_category,
            'is_same_grid': is_same_grid,
            'is_adjacent': is_adjacent,
            'context': context  # 呼び出しコンテキストを追加
        }
        
        self.travel_time_logs.append(log_entry)
        if distance_category in self.distance_categories:
            self.distance_categories[distance_category].append(travel_time_seconds / 60.0)
    
    def _check_adjacent_grids(self, h3_1: str, h3_2: str) -> bool:
        """H3グリッドが隣接しているかチェック（簡易版）"""
        try:
            neighbors = h3.grid_disk(h3_1, 1)
            return h3_2 in neighbors
        except:
            return False
    
    def save_detailed_logs(self):
        """詳細ログをCSVファイルに保存"""
        csv_path = os.path.join(self.output_dir, 'travel_time_detailed_logs.csv')
        
        if self.travel_time_logs:
            fieldnames = self.travel_time_logs[0].keys()
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.travel_time_logs)
            
            print(f"移動時間詳細ログを保存しました: {csv_path}")
    
    def generate_analysis_report(self):
        """分析レポートを生成"""
        if not self.travel_time_logs:
            print("分析対象のログデータがありません")
            return
        
        # 基本統計の計算
        df = pd.DataFrame(self.travel_time_logs)
        
        print("\n=== 移動時間詳細分析レポート ===")
        
        # 0. コンテキスト別分析
        print(f"\n0. コンテキスト（呼び出し元）別分析:")
        context_counts = df['context'].value_counts()
        for context, count in context_counts.items():
            print(f"   {context}: {count}回")
        
        # 1. 同一グリッド移動の分析
        same_grid_df = df[df['is_same_grid'] == True]
        print(f"\n1. 同一グリッド内移動の分析:")
        print(f"   総移動回数: {len(df)}")
        print(f"   同一グリッド移動回数: {len(same_grid_df)} ({len(same_grid_df)/len(df)*100:.1f}%)")
        
        for phase in ['response', 'transport', 'return']:
            phase_same = same_grid_df[same_grid_df['phase'] == phase]
            phase_total = df[df['phase'] == phase]
            if len(phase_total) > 0:
                percentage = len(phase_same) / len(phase_total) * 100
                avg_time = phase_same['travel_time_minutes'].mean() if len(phase_same) > 0 else 0
                print(f"   {phase}フェーズ: {len(phase_same)}/{len(phase_total)} ({percentage:.1f}%) 平均時間: {avg_time:.1f}分")
        
        # 2. 距離カテゴリ別分析
        print(f"\n2. 距離カテゴリ別分析:")
        for category, times in self.distance_categories.items():
            if times:
                avg_time = np.mean(times)
                count = len(times)
                print(f"   {category}: {count}回, 平均時間: {avg_time:.1f}分")
        
        # 3. フェーズ別統計
        print(f"\n3. フェーズ別移動時間統計:")
        for phase in ['response', 'transport', 'return']:
            phase_df = df[df['phase'] == phase]
            if len(phase_df) > 0:
                stats = {
                    'count': len(phase_df),
                    'mean': phase_df['travel_time_minutes'].mean(),
                    'median': phase_df['travel_time_minutes'].median(),
                    'std': phase_df['travel_time_minutes'].std(),
                    'min': phase_df['travel_time_minutes'].min(),
                    'max': phase_df['travel_time_minutes'].max()
                }
                print(f"   {phase}: 平均{stats['mean']:.1f}分, 中央値{stats['median']:.1f}分, 標準偏差{stats['std']:.1f}分")
                print(f"           最小{stats['min']:.1f}分, 最大{stats['max']:.1f}分, 件数{stats['count']}")
        
        # 4. 可視化
        self._create_visualizations(df)
    
    def _create_visualizations(self, df):
        """移動時間分析の可視化"""
        
        # 1. 距離カテゴリ別移動時間分布
        plt.figure(figsize=(15, 10))
        
        # 1-1. 距離カテゴリ別ヒストグラム
        plt.subplot(2, 3, 1)
        categories = ['same_grid', 'adjacent_grid', 'near_distance', 'medium_distance', 'far_distance']
        category_labels = ['同一グリッド', '隣接グリッド', '近距離', '中距離', '遠距離']
        
        for i, (cat, label) in enumerate(zip(categories, category_labels)):
            if self.distance_categories[cat]:
                plt.hist(self.distance_categories[cat], bins=20, alpha=0.6, label=label)
        
        plt.xlabel('移動時間（分）', fontfamily='Meiryo')
        plt.ylabel('頻度', fontfamily='Meiryo')
        plt.title('距離カテゴリ別移動時間分布', fontfamily='Meiryo')
        plt.legend(prop={'family': 'Meiryo'})
        plt.grid(True, alpha=0.3)
        
        # 1-2. フェーズ別移動時間箱ひげ図
        plt.subplot(2, 3, 2)
        phases = ['response', 'transport', 'return']
        phase_data = [df[df['phase'] == phase]['travel_time_minutes'].tolist() for phase in phases]
        
        plt.boxplot(phase_data, labels=['出動', '搬送', '帰署'])
        plt.ylabel('移動時間（分）', fontfamily='Meiryo')
        plt.title('フェーズ別移動時間分布', fontfamily='Meiryo')
        plt.grid(True, alpha=0.3)
        
        # 1-3. 同一グリッド vs その他の比較
        plt.subplot(2, 3, 3)
        same_grid_times = df[df['is_same_grid'] == True]['travel_time_minutes']
        other_times = df[df['is_same_grid'] == False]['travel_time_minutes']
        
        plt.hist(same_grid_times, bins=20, alpha=0.6, label=f'同一グリッド (n={len(same_grid_times)})')
        plt.hist(other_times, bins=20, alpha=0.6, label=f'異なるグリッド (n={len(other_times)})')
        plt.xlabel('移動時間（分）', fontfamily='Meiryo')
        plt.ylabel('頻度', fontfamily='Meiryo')
        plt.title('同一グリッド移動 vs その他', fontfamily='Meiryo')
        plt.legend(prop={'family': 'Meiryo'})
        plt.grid(True, alpha=0.3)
        
        # 1-4. 距離と移動時間の散布図
        plt.subplot(2, 3, 4)
        valid_distance_df = df[df['distance_km'].notna()]
        if len(valid_distance_df) > 0:
            plt.scatter(valid_distance_df['distance_km'], valid_distance_df['travel_time_minutes'], alpha=0.6)
            plt.xlabel('移動距離（km）', fontfamily='Meiryo')
            plt.ylabel('移動時間（分）', fontfamily='Meiryo')
            plt.title('距離 vs 移動時間', fontfamily='Meiryo')
            plt.grid(True, alpha=0.3)
        
        # 1-5. フェーズ別同一グリッド移動の割合
        plt.subplot(2, 3, 5)
        phase_same_ratios = []
        phase_labels = []
        
        for phase in phases:
            phase_df = df[df['phase'] == phase]
            if len(phase_df) > 0:
                same_ratio = len(phase_df[phase_df['is_same_grid'] == True]) / len(phase_df) * 100
                phase_same_ratios.append(same_ratio)
                phase_labels.append(phase)
        
        plt.bar(phase_labels, phase_same_ratios)
        plt.ylabel('同一グリッド移動の割合（%）', fontfamily='Meiryo')
        plt.title('フェーズ別同一グリッド移動率', fontfamily='Meiryo')
        plt.grid(True, alpha=0.3)
        
        # 1-6. 移動時間の累積分布
        plt.subplot(2, 3, 6)
        for phase in phases:
            phase_times = df[df['phase'] == phase]['travel_time_minutes'].sort_values()
            if len(phase_times) > 0:
                cumulative = np.arange(1, len(phase_times) + 1) / len(phase_times) * 100
                plt.plot(phase_times, cumulative, label=phase)
        
        plt.xlabel('移動時間（分）', fontfamily='Meiryo')
        plt.ylabel('累積確率（%）', fontfamily='Meiryo')
        plt.title('移動時間の累積分布', fontfamily='Meiryo')
        plt.legend(prop={'family': 'Meiryo'})
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'travel_time_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"移動時間分析グラフを保存しました: {os.path.join(self.output_dir, 'travel_time_analysis.png')}")

class ValidationSimulator:
    """検証用シミュレータ"""
    
    def __init__(self, 
                 travel_time_matrices: Dict[str, np.ndarray],
                 travel_distance_matrices: Dict[str, np.ndarray],  # 移動距離行列を追加
                 grid_mapping: Dict,
                 service_time_generator: ServiceTimeGenerator,
                 hospital_h3_indices: List[str],
                 hospital_data: Optional[pd.DataFrame] = None,
                 use_probabilistic_selection: bool = True,  # 軽症・中等症・死亡の確率的選択制御
                 enable_breaks: bool = True,  # 休憩機能の有効/無効を追加
                 dispatch_strategy: str = 'closest',  # 追加
                 strategy_config: Dict = None):  # 追加
        
        self.travel_time_matrices = travel_time_matrices
        self.travel_distance_matrices = travel_distance_matrices  # 移動距離行列を追加
        
        if 'default' in self.travel_distance_matrices and len(self.travel_distance_matrices) == 1:
            default_matrix = self.travel_distance_matrices['default']
            self.travel_distance_matrices = {
                'dispatch_to_scene': default_matrix,
                'scene_to_hospital': default_matrix,
                'hospital_to_station': default_matrix
            }

        self.grid_mapping = grid_mapping
        self.service_time_generator = service_time_generator
        self.hospital_h3_indices = hospital_h3_indices
        self.enable_breaks = enable_breaks  # 休憩機能フラグ
        
        # シミュレーション状態
        self.current_time = 0.0
        self.event_queue = []
        self.ambulances = {}
        self.calls: Dict[str, EmergencyCall] = {}
        self.configured_end_time = 0.0

        # ログ出力設定
        self.verbose_logging = True
        
        # 移動時間分析機能を追加
        self.travel_time_analyzer = None
        self.enable_travel_time_analysis = False
        
        # 実際の移動時間のみを記録するためのフラグ
        self.is_actual_movement = False
        self.current_movement_context = 'unknown'
        
        # 統計情報
        self.statistics = {
            'total_calls': 0,
            'completed_calls': 0,
            'response_times': [],
            'response_times_by_severity': {},
            'utilization_by_hour': {h: [] for h in range(24)},
            'calls_by_hour': {h: 0 for h in range(24)},
            'threshold_6min': {'total': 0, 'achieved': 0},
            'threshold_13min': {'total': 0, 'achieved': 0},
            'threshold_6min_by_severity': {},
            'threshold_13min_by_severity': {},
            'ambulance_utilization': {},
            'queue_lengths': [],
            'total_activity_times': [],
            'phase_durations': {
                'dispatch_to_scene': [],
                'on_scene': [],
                'scene_to_hospital': [],
                'at_hospital': [],
                'hospital_to_station': []
            },
            'phase_durations_by_severity': {},
            'travel_time_default_usage_count': 0,
            'hospital_selection_stats': {
                'tertiary_selections': 0,
                'secondary_primary_selections': 0,
                'no_hospital_found': 0,
                'by_severity': {},
                'selection_methods': {  # 新規追加
                    'probabilistic_success': 0,
                    'deterministic_fallback': 0,
                    'static_fallback_used': 0,
                    'error_fallback': 0
                }
            },
            # 移動距離の統計を追加
            'travel_distances': {
                'dispatch_to_scene': [],
                'scene_to_hospital': [],
                'hospital_to_station': []
            },
            'travel_distances_by_severity': {},
            'total_distance': 0.0,
            # 休憩関連統計を追加
            'rest_time_stats': {
                'total_count': 0,
                'by_reason': {
                    'lunch_period': 0,
                    'lunch_extended': 0,
                    'dinner_period': 0,
                    'dinner_extended': 0
                },
                'by_section': {i: 0 for i in range(1, 11)},
                'by_hour': {h: 0 for h in range(24)}
            },
            'interval_stats': {
                'total_count': 0,
                'by_hour': {h: 0 for h in range(24)},
                'average_activity_before_interval': []
            },
            'crew_changes': {
                'total_count': 0,
                'by_hour': {h: 0 for h in range(24)}
            }
        }
        
        # 病院データの処理と分類 (statistics 初期化後に呼び出し)
        self.hospital_data = hospital_data
        self.tertiary_hospitals = set()  # 3次救急医療機関
        self.secondary_primary_hospitals = set()  # 2次以下の医療機関
        self._classify_hospitals()
        
        # ディスパッチ戦略の初期化を追加
        self.dispatch_strategy = StrategyFactory.create_strategy(
            dispatch_strategy, 
            strategy_config or {}
        )
        
        # コンテキスト情報の初期化を追加
        self.dispatch_context = DispatchContext()
        self.dispatch_context.grid_mapping = self.grid_mapping
        self.dispatch_context.all_h3_indices = set(grid_mapping.keys())
        
        # 確率的病院選択モデルの初期化（軽症・中等症・死亡のみ、病院データ処理後に追加）
        self.use_probabilistic_selection = use_probabilistic_selection
        self.hospital_selection_model = None
        self.default_hospital_probs = None
        self.hospital_acceptance_stats = None
        
        if self.use_probabilistic_selection:
            self._load_hospital_selection_model()
    
    def _classify_hospitals(self):
        """病院を3次救急とそれ以外に分類"""
        if self.hospital_data is None:
            print("警告: 病院データが提供されていません。全ての病院を2次以下として扱います。")
            self.secondary_primary_hospitals = set(self.hospital_h3_indices)
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
            unclassified = set(self.hospital_h3_indices) - self.tertiary_hospitals - self.secondary_primary_hospitals
            if unclassified:
                print(f"  未分類病院（2次以下に追加）: {len(unclassified)}件")
                self.secondary_primary_hospitals.update(unclassified)
        else:
            print("警告: hospital_dataに'genre_code'カラムが見つかりません。全ての病院を2次以下として扱います。")
            self.secondary_primary_hospitals = set(self.hospital_h3_indices)
        
        # 統計初期化
        for severity in ['軽症', '中等症', '重症', '重篤', '死亡', 'その他']:
            self.statistics['hospital_selection_stats']['by_severity'][severity] = {
                'tertiary': 0,
                'secondary_primary': 0,
                'default': 0,
                'probabilistic_success': 0,
                'deterministic_fallback': 0,
                'static_fallback_used': 0,
                'error_fallback': 0
            }
    
    def _load_hospital_selection_model(self):
        """確率的病院選択モデルを読み込む"""
        # ★変更点1：読み込むモデルファイルを新しい「改訂版」に変更
        data_dir = CURRENT_PROJECT_DIR / "data" / "tokyo" / "processed"
        model_path = data_dir / "hospital_selection_model_revised.pkl"
        
        try:
            with open(model_path, 'rb') as f:
                main_model = pickle.load(f)
                self.hospital_selection_model = main_model['selection_probabilities']
                
                # ★変更点2：「静的フォールバックモデル」を読み込む
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
    
    
    def enable_detailed_analysis(self, output_dir: str):
        """詳細分析機能を有効にする"""
        self.enable_travel_time_analysis = True
        self.travel_time_analyzer = TravelTimeAnalyzer(output_dir)
        print("移動時間詳細分析機能を有効にしました")

    # 休憩関連のメソッドを追加
    def _calculate_section_dispatch_rate(self, section: int) -> float:
        """指定された方面の救急隊出場率を計算"""
        section_ambulances = [amb for amb in self.ambulances.values() if amb.section == section]
        if not section_ambulances:
            return 0.0
        
        dispatched_count = sum(1 for amb in section_ambulances 
                             if amb.status != AmbulanceStatus.AVAILABLE or amb.is_on_break)
        return dispatched_count / len(section_ambulances)
    
    def _check_rest_time_requirements(self, ambulance: Ambulance, current_time: float) -> str:
        """レストタイム取得要件をチェックし、取得理由を返す"""
        hour = int((current_time / 3600) % 24)
        section_dispatch_rate = self._calculate_section_dispatch_rate(ambulance.section)
        
        # 方面出場率の閾値判定
        if hour < 23:
            dispatch_threshold = 0.8
        else:
            dispatch_threshold = 0.9
        
        # ①昼食時間帯（11-13時）のチェック
        if hour >= 13 and ambulance.lunch_period_standby_time < 1800:  # 30分未満
            if section_dispatch_rate < dispatch_threshold:
                return "lunch_period"
        
        # ②昼食延長時間帯（11-15時）のチェック
        if hour >= 15 and ambulance.lunch_extended_standby_time < 1800:
            return "lunch_extended"
        
        # ③夕食時間帯（17-19時）のチェック
        if hour >= 19 and ambulance.dinner_period_standby_time < 1800:
            if section_dispatch_rate < dispatch_threshold:
                return "dinner_period"
        
        # ④夕食延長時間帯（17-20時）のチェック
        if hour >= 20 and ambulance.dinner_extended_standby_time < 1800:
            return "dinner_extended"
        
        return ""  # 休憩不要
    
    def _check_interval_requirements(self, ambulance: Ambulance, current_time: float) -> bool:
        """出場間インターバル取得要件をチェック"""
        hour = int((current_time / 3600) % 24)
        
        # 深夜時間帯（22時～5時59分）のチェック
        if hour >= 22 or hour < 6:
            # 4時間または3件を超えた場合
            if ambulance.night_activity_duration > 14400 or ambulance.night_activity_count >= 3:
                return True
        
        return False
    
    def _update_standby_time(self, ambulance: Ambulance, start_time: float, end_time: float):
        """指定期間の待機時間を更新"""
        if ambulance.status != AmbulanceStatus.AVAILABLE or ambulance.is_on_break:
            return
        
        # 1分刻みで時間帯を確認
        for t in range(int(start_time), int(end_time), 60):
            if t >= end_time:
                break
            
            hour = int((t / 3600) % 24)
            minute_duration = min(60, end_time - t)
            
            if 11 <= hour < 13:
                ambulance.lunch_period_standby_time += minute_duration
            if 11 <= hour < 15:
                ambulance.lunch_extended_standby_time += minute_duration
            if 17 <= hour < 19:
                ambulance.dinner_period_standby_time += minute_duration
            if 17 <= hour < 20:
                ambulance.dinner_extended_standby_time += minute_duration
    
    def _should_change_crew(self, ambulance: Ambulance, current_time: float) -> bool:
        """乗務員交代が必要かチェック（8:30のみ交代）"""
        hour = int((current_time / 3600) % 24)
        minute = int((current_time / 60) % 60)
        
        # 8:30以降かつ、今日まだ交代していない場合
        if hour == 8 and minute >= 30:
            if ambulance.last_crew_change_time is None:
                return True
            
            # 最後の交代から20時間以上経過している場合（前日の交代）
            time_since_last_change = current_time - ambulance.last_crew_change_time
            if time_since_last_change > 72000:  # 20時間
                return True
        
        return False
    
    def _handle_crew_change(self, ambulance: Ambulance, current_time: float):
        """乗務員交代による状態リセット（8:30の交代）"""
        ambulance.last_crew_change_time = current_time
        
        # 休憩必要性のリセット
        ambulance.lunch_period_standby_time = 0.0
        ambulance.lunch_extended_standby_time = 0.0
        ambulance.dinner_period_standby_time = 0.0
        ambulance.dinner_extended_standby_time = 0.0
        ambulance.last_rest_time = None
        
        # 深夜活動カウンタもリセット
        ambulance.night_activity_duration = 0.0
        ambulance.night_activity_count = 0
        ambulance.night_activity_start_time = None
        
        # 統計情報の更新
        self.statistics['crew_changes']['total_count'] += 1
        hour = int((current_time / 3600) % 24)
        self.statistics['crew_changes']['by_hour'][hour] += 1
        
        if self.verbose_logging:
            print(f"[INFO] 救急車 {ambulance.id} の乗務員が交代しました (時刻: {current_time/3600:.1f}時)")
    
    def _start_break(self, ambulance: Ambulance, break_type: str, duration_minutes: int, reason: str):
        """休憩を開始（理由付き）"""
        ambulance.is_on_break = True
        ambulance.status = AmbulanceStatus.ON_BREAK
        ambulance.break_end_time = self.current_time + (duration_minutes * 60)
        ambulance.break_type = break_type
        
        # 統計情報を更新
        hour = int((self.current_time / 3600) % 24)
        
        if break_type == 'rest_time':
            self.statistics['rest_time_stats']['total_count'] += 1
            self.statistics['rest_time_stats']['by_reason'][reason] += 1
            self.statistics['rest_time_stats']['by_section'][ambulance.section] += 1
            self.statistics['rest_time_stats']['by_hour'][hour] += 1
        else:  # interval
            self.statistics['interval_stats']['total_count'] += 1
            self.statistics['interval_stats']['by_hour'][hour] += 1
            self.statistics['interval_stats']['average_activity_before_interval'].append(
                ambulance.night_activity_duration / 3600  # 時間単位で記録
            )
        
        # 休憩終了イベントをスケジュール
        break_end_event = Event(
            time=ambulance.break_end_time,
            event_type=EventType.BREAK_END,
            data={'ambulance_id': ambulance.id}
        )
        heapq.heappush(self.event_queue, break_end_event)
        
        if self.verbose_logging:
            print(f"[BREAK] 救急車 {ambulance.id} が休憩開始: {break_type} ({reason}), "
                  f"{duration_minutes}分間, 終了予定: {ambulance.break_end_time/3600:.1f}時")
    
    def _end_break(self, ambulance: Ambulance):
        """休憩を終了"""
        ambulance.is_on_break = False
        ambulance.status = AmbulanceStatus.AVAILABLE
        ambulance.break_end_time = None
        
        if ambulance.break_type == 'rest_time':
            ambulance.last_rest_time = self.current_time
        elif ambulance.break_type == 'interval':
            # 深夜活動カウンタをリセット
            ambulance.night_activity_duration = 0.0
            ambulance.night_activity_count = 0
        
        ambulance.break_type = None
        
        if self.verbose_logging:
            print(f"[BREAK] 救急車 {ambulance.id} が休憩終了、利用可能になりました")

    def get_travel_time_for_actual_movement(self, from_h3: str, to_h3: str, phase: str, context: str) -> float:
        """実際の移動時の移動時間取得（分析ログ対象）"""
        self.is_actual_movement = True
        self.current_movement_context = context
        travel_time = self.get_travel_time(from_h3, to_h3, phase)
        self.is_actual_movement = False
        self.current_movement_context = 'unknown'
        return travel_time

    def initialize_ambulances(self, 
                              ambulance_data: pd.DataFrame, 
                              include_daytime_ambulances: bool = True,
                              initial_active_rate_range: Tuple[float, float] = (0.0, 0.0),
                              initial_availability_time_range_minutes: Tuple[int, int] = (0, 0)
                             ):
        """
        救急車の初期化。一部をランダムに「初期利用不能」として設定し、指定時間後に利用可能にする。
        """
        
        print("\n--- 救急車初期化開始 ---")
        print(f"デイタイム救急を含めるか: {include_daytime_ambulances}")
        print(f"初期利用不能隊の割合範囲: {initial_active_rate_range[0]*100:.0f}% - {initial_active_rate_range[1]*100:.0f}%")
        print(f"初期利用不能隊の利用可能までの時間範囲: {initial_availability_time_range_minutes[0]}分 - {initial_availability_time_range_minutes[1]}分")

        if ambulance_data.empty:
            print("警告: 救急署データが空のため、救急車の初期化は行われません。")
            self.ambulances = {}
            return

        original_station_count = len(ambulance_data)
        
        if 'team_name' in ambulance_data.columns:
            ambulance_data = ambulance_data[ambulance_data['team_name'] != '救急隊なし'].copy()
            if len(ambulance_data) < original_station_count:
                print(f"情報: 'team_name' が '救急隊なし' の {original_station_count - len(ambulance_data)} 件の救急署データを除外しました。")

        if not include_daytime_ambulances and 'team_name' in ambulance_data.columns:
            original_count_before_daytime_filter = len(ambulance_data)
            ambulance_data = ambulance_data[~ambulance_data['team_name'].astype(str).str.contains("デイタイム救急")].copy()
            if len(ambulance_data) < original_count_before_daytime_filter:
                print(f"情報: 'team_name' に 'デイタイム救急' を含む {original_count_before_daytime_filter - len(ambulance_data)} 件の救急署データを除外しました。")
            
        if ambulance_data.empty:
            print("警告: フィルタリングの結果、初期化対象の救急署データが0件になりました。救急車は初期化されません。")
            self.ambulances = {}
            return

        all_initialized_ambulances: List[Ambulance] = []
        for index, row in ambulance_data.iterrows():
            h3_index = h3.latlng_to_cell(row['latitude'], row['longitude'], 9)
            
            # sectionカラムの確認と読み込み
            section = row.get('section', 1)  # デフォルトは1
            if not (1 <= section <= 10):
                print(f"警告: 無効なsection値 {section} が検出されました。デフォルト値1を使用します。")
                section = 1
            
            num_ambulances_to_create = 0
            if 'amb' in row and pd.notna(row['amb']):
                try:
                    amb_value = int(float(str(row['amb'])))
                    if amb_value > 0:
                        num_ambulances_to_create = 1 
                    else:
                        continue
                except ValueError:
                    continue
            else:
                continue

            if num_ambulances_to_create <= 0:
                 continue

            team_name = row.get('team_name', f"Station_{h3_index}")
            if not team_name: 
                team_name = f"Station_{h3_index}"

            for i in range(num_ambulances_to_create):
                amb_id = f"{team_name}_{i}"
                # 全ての救急車を一度 AVAILABLE として基本情報を設定
                ambulance = Ambulance(
                    id=amb_id,
                    team_name=team_name,
                    station_h3_index=h3_index,
                    current_h3_index=h3_index,
                    section=section,  # 方面番号を設定
                    status=AmbulanceStatus.AVAILABLE # 初期状態はAVAILABLE
                )
                self.ambulances[amb_id] = ambulance
                all_initialized_ambulances.append(ambulance)
                self.statistics['ambulance_utilization'][amb_id] = {
                    'active_time': 0.0,
                    'calls_handled': 0
                }
        
        print(f"一時初期化完了: {len(self.ambulances)}台の救急車 (全て利用可能状態)")

        # 初期利用不能隊の設定
        if self.ambulances and initial_active_rate_range[1] > 0:
            num_total_ambulances = len(all_initialized_ambulances)
            actual_active_rate = random.uniform(initial_active_rate_range[0], initial_active_rate_range[1])
            num_to_set_unavailable = int(num_total_ambulances * actual_active_rate)
            
            if num_to_set_unavailable > 0:
                print(f"  うち {num_to_set_unavailable}台 ({actual_active_rate*100:.1f}%) の救急車を初期利用不能として設定します...")
                
                unavailable_ambulances_sample = random.sample(all_initialized_ambulances, k=min(num_to_set_unavailable, num_total_ambulances))
                
                for amb in unavailable_ambulances_sample:
                    # ステータスを一時的に利用不能を示すものに変更
                    # (AMBULANCE_AVAILABLEイベントでAVAILABLEに戻ることを期待)
                    amb.status = AmbulanceStatus.RETURNING # 便宜的に「帰署中」を利用不能状態として使用
                    
                    min_avail_time_sec = initial_availability_time_range_minutes[0] * 60
                    max_avail_time_sec = initial_availability_time_range_minutes[1] * 60
                    time_to_become_available_sec = random.uniform(min_avail_time_sec, max_avail_time_sec)
                    
                    available_event_time = self.current_time + time_to_become_available_sec 
                    
                    available_event = Event(
                        time=available_event_time,
                        event_type=EventType.AMBULANCE_AVAILABLE,
                        data={'call_id': f"initial_unavailable_{amb.id}", 'ambulance_id': amb.id}
                    )
                    heapq.heappush(self.event_queue, available_event)
                    
                    # if self.verbose_logging:
                    #     print(f"    - 救急車 {amb.id}: 初期ステータスを {amb.status.value} (利用不能) に設定。")
                    #     print(f"      {available_event_time/60:.1f}分後 ({available_event_time:.0f}秒後) に利用可能になるイベントを投入。")
            else:
                print("  初期利用不能として設定する救急車はありません。")
        else:
            print("  初期利用不能隊の設定はスキップされました（割合0%または対象救急車なし）。")


        print(f"最終的な初期化救急車台数: {len(self.ambulances)}台")
        available_count_final = sum(1 for amb in self.ambulances.values() if amb.status == AmbulanceStatus.AVAILABLE)
        unavailable_initial_count_final = len(self.ambulances) - available_count_final
        print(f"  うち初期利用可能: {available_count_final}台")
        print(f"  うち初期利用不能 (指定時間後に利用可能予定): {unavailable_initial_count_final}台")
        print("--- 救急車初期化完了 ---")
        
    
    def add_emergency_calls(self, calls_df: pd.DataFrame, sim_start_datetime: datetime):
        """救急事案をイベントキューに追加"""
        # if self.verbose_logging:
        #     print(f"[INFO] Adding {len(calls_df)} emergency calls to event queue. Sim start: {sim_start_datetime}")
        
        # # 東京23区のデータのみをフィルタリング
        # calls_df = calls_df[calls_df['special_flag'] == 1].copy()
        # if calls_df.empty:
        #     print("警告: 東京23区の救急要請データが見つかりません")
        #     return
        
        # print(f"東京23区の救急要請データ: {len(calls_df)}件")
        
        # 日時を一括で変換
        call_times = (pd.to_datetime(calls_df['出場年月日時分']) - sim_start_datetime).dt.total_seconds()
        
        # H3インデックスを一括で計算
        h3_indices = [h3.latlng_to_cell(row['Y_CODE'], row['X_CODE'], 9) 
                     for _, row in calls_df.iterrows()]
        
        # イベントを一括で作成
        events = [
            Event(
                time=call_time,
                event_type=EventType.NEW_CALL,
                data={
                    'call_id': str(call_row['救急事案番号キー']),
                    'h3_index': h3_idx,
                    'severity': call_row.get('収容所見程度', 'その他'),
                    'team_name': call_row.get('隊名', None),
                    'call_datetime': call_row['出場年月日時分']
                }
            )
            for call_time, h3_idx, (_, call_row) in zip(call_times, h3_indices, calls_df.iterrows())
        ]
        
        # イベントキューに一括で追加
        for event in events:
            heapq.heappush(self.event_queue, event)
    
    def get_travel_time(self, from_h3: str, to_h3: str, phase: str) -> float:
        """移動時間の取得（秒単位）"""
        from_idx = self.grid_mapping.get(from_h3)
        to_idx = self.grid_mapping.get(to_h3)
        
        if from_idx is None or to_idx is None:
            if self.verbose_logging:
                print(f"警告: 移動時間が見つかりません ({from_h3} または {to_h3} がgrid_mappingにありません, phase={phase}). デフォルト値300秒を使用します。")
            self.statistics['travel_time_default_usage_count'] += 1
            return 300.0

        current_travel_time_matrix = self.travel_time_matrices.get(phase)
        
        # 指定されたフェーズの行列がない場合、'response'フェーズの行列で代用
        if current_travel_time_matrix is None:
            if self.verbose_logging:
                print(f"警告: '{phase}'フェーズの移動時間行列が見つかりません。'response'行列をフォールバックとして使用します。")
            current_travel_time_matrix = self.travel_time_matrices.get('response')
            
            # それもなければデフォルト値を返す
            if current_travel_time_matrix is None:
                if self.verbose_logging:
                    print(f"エラー: フォールバック用の'response'行列も見つかりません。デフォルト値300秒を使用します。")
                self.statistics['travel_time_default_usage_count'] += 1
                return 300.0
        
        travel_time = current_travel_time_matrix[from_idx, to_idx]
        
        # 詳細分析機能が有効で、かつ実際の移動時のみログを記録
        if (self.enable_travel_time_analysis and self.travel_time_analyzer and 
            self.is_actual_movement):
            # 距離情報も取得
            distance_km = None
            try:
                distance_km = self.get_travel_distance(from_h3, to_h3, phase)
            except:
                pass
            
            self.travel_time_analyzer.log_travel_time(
                from_h3=from_h3,
                to_h3=to_h3,
                phase=phase,
                travel_time_seconds=travel_time,
                distance_km=distance_km,
                context=self.current_movement_context
            )
            
            # 同一グリッド移動の詳細ログ
            if from_h3 == to_h3 and self.verbose_logging:
                print(f"[SAME_GRID] Phase: {phase}, Time: {travel_time/60:.1f}分, From/To: {from_h3}, Context: {self.current_movement_context}")
        
        return travel_time

    def get_travel_distance(self, from_h3: str, to_h3: str, phase: str) -> float:
        """指定された2点間の移動距離を取得（キロメートル単位）"""
        try:
            # phaseの変換
            phase_map = {
                'response': 'dispatch_to_scene',
                'transport': 'scene_to_hospital',
                'return': 'hospital_to_station'
            }
            matrix_phase = phase_map.get(phase, phase)
            
            from_idx = self.grid_mapping[from_h3]
            to_idx = self.grid_mapping[to_h3]
            
            distance = self.travel_distance_matrices[matrix_phase][from_idx, to_idx]
            
            if self.verbose_logging:
                print(f"[DEBUG] 移動距離取得: {from_h3} -> {to_h3}, phase={matrix_phase}, indices=({from_idx}, {to_idx}), distance={distance:.1f}km")
            
            return distance
        except KeyError as e:
            if self.verbose_logging:
                print(f"警告: 移動距離が見つかりません ({from_h3} または {to_h3} がgrid_mappingにありません, phase={phase})")
                print(f"エラー詳細: {str(e)}")
                print(f"利用可能な距離行列: {list(self.travel_distance_matrices.keys())}")
            return 0.0  # デフォルト値
    
    def find_closest_available_ambulance(self, call_h3: str, severity: str = None) -> Optional[Ambulance]:
        """
        ディスパッチ戦略を使用して最適な救急車を選択
        
        Args:
            call_h3: 事案発生地点のH3インデックス
            severity: 傷病度（オプション）
        
        Returns:
            選択された救急車オブジェクト、またはNone
        """
        # 利用可能な救急車を取得（休憩中を除外）
        available_ambulances = [
            amb for amb in self.ambulances.values() 
            if amb.status == AmbulanceStatus.AVAILABLE and not amb.is_on_break
        ]
        
        if not available_ambulances:
            if self.verbose_logging:
                print(f"[INFO] No available ambulances for call at {call_h3} at time {self.current_time:.2f}")
            return None
        
        # AmbulanceInfoオブジェクトのリストを作成
        ambulance_infos = []
        for amb in available_ambulances:
            amb_info = AmbulanceInfo(
                id=amb.id,
                current_h3=amb.current_h3_index,
                station_h3=amb.station_h3_index,
                status=amb.status.value,
                total_calls_today=amb.num_calls_handled,
                current_workload=0.0  # 必要に応じて計算
            )
            ambulance_infos.append(amb_info)
        
        # EmergencyRequestオブジェクトを作成
        priority = self.dispatch_strategy.get_severity_priority(severity) if severity else DispatchPriority.LOW
        request = EmergencyRequest(
            id=f"temp_{self.current_time}",  # 仮ID
            h3_index=call_h3,
            severity=severity or "その他",
            time=self.current_time,
            priority=priority
        )
        
        # DispatchContextを更新
        self.dispatch_context.current_time = self.current_time
        self.dispatch_context.hour_of_day = int((self.current_time / 3600) % 24)
        self.dispatch_context.total_ambulances = len(self.ambulances)
        self.dispatch_context.available_ambulances = len(available_ambulances)
        
        # ★★★ PPO戦略用：全救急車の状態情報を設定 ★★★
        self.dispatch_context.all_ambulances = {}
        for amb_id, ambulance in self.ambulances.items():
            self.dispatch_context.all_ambulances[amb_id] = ambulance
        
        # 戦略を使用して救急車を選択
        selected_info = self.dispatch_strategy.select_ambulance(
            request=request,
            available_ambulances=ambulance_infos,
            travel_time_func=self.get_travel_time,
            context=self.dispatch_context
        )
        
        # ★★★ 直近隊選択統計を記録（全戦略共通） ★★★
        if selected_info and hasattr(self.dispatch_strategy, '_record_dispatch_statistics'):
            try:
                self.dispatch_strategy._record_dispatch_statistics(
                    selected_info,
                    request,
                    ambulance_infos,
                    self.get_travel_time
                )
            except Exception:
                # エラーが発生してもシミュレーションは継続
                pass
        
        if selected_info:
            # AmbulanceInfoからAmbulanceオブジェクトを取得
            selected_ambulance = next(
                (amb for amb in available_ambulances if amb.id == selected_info.id),
                None
            )
            
            if selected_ambulance and self.verbose_logging:
                travel_time = self.get_travel_time(
                    selected_ambulance.current_h3_index, 
                    call_h3, 
                    phase='response'
                )
                print(f"[INFO] Call at {call_h3} (severity: {severity}): "
                      f"Selected ambulance {selected_ambulance.id} at {selected_ambulance.current_h3_index}. "
                      f"Est. travel time: {travel_time:.2f}s. "
                      f"Strategy: {self.dispatch_strategy.name}")
            
            return selected_ambulance
        
        return None
    
    def _select_hospital_deterministic(self, incident_h3: str, severity: str) -> Optional[str]:
        """決定論的な病院選択"""
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
                    
                    # 統計情報を更新する
                    self.statistics['hospital_selection_stats']['tertiary_selections'] += 1
                    self._update_hospital_selection_stats(severity, 'tertiary', 'deterministic_fallback')
                    if self.verbose_logging:
                        print(f"[INFO] {severity}: 3次救急を選択 {selected}")
                        
                    return selected
        
        # 軽症・中等症・死亡の場合、または3次救急が見つからない重症・重篤ケース：2次以下から探す
        if self.secondary_primary_hospitals:
            nearest_secondary = self._find_nearest_hospital(incident_h3, self.secondary_primary_hospitals)
            if nearest_secondary:
                self.statistics['hospital_selection_stats']['secondary_primary_selections'] += 1
                self._update_hospital_selection_stats(severity, 'secondary_primary', 'deterministic_fallback')
                if self.verbose_logging:
                    selection_reason = "2次以下優先" if severity not in severe_conditions else "3次救急見つからず2次以下で代用"
                    print(f"[INFO] {severity}: {selection_reason} {nearest_secondary}")
                return nearest_secondary

        # それでも見つからない場合：軽症・中等症・死亡なら3次から探す
        if severity not in severe_conditions and self.tertiary_hospitals:
            nearest_tertiary = self._find_nearest_hospital(incident_h3, self.tertiary_hospitals)
            if nearest_tertiary:
                self.statistics['hospital_selection_stats']['tertiary_selections'] += 1
                self._update_hospital_selection_stats(severity, 'tertiary', 'deterministic_fallback')
                if self.verbose_logging:
                    print(f"[INFO] {severity}: 2次以下見つからず3次救急で代用 {nearest_tertiary}")
                return nearest_tertiary
                
        # 全ての候補を探しても見つからない場合
        self.statistics['hospital_selection_stats']['no_hospital_found'] += 1
        self._update_hospital_selection_stats(severity, 'no_hospital_found', 'error_fallback')
        if self.verbose_logging:
            print(f"[WARN] {severity}: 病院が見つかりませんでした")
        return None
    
    def select_hospital(self, incident_h3: str, severity: str) -> Optional[str]:
        """傷病度に応じた病院選択（★静的フォールバックモデルを使用する改訂版）"""
        
        severe_conditions = ['重症', '重篤']
        
        # 重症・重篤の案件は決定論的選択（変更なし）
        if severity in severe_conditions:
            return self._select_hospital_deterministic(incident_h3, severity)
        
        # 軽症・中等症・死亡：確率的選択
        if not self.use_probabilistic_selection:
            return self._select_hospital_deterministic(incident_h3, severity)

        # 時間情報とキーの作成（変更なし）
        current_hour = int((self.current_time / 3600) % 24)
        time_slot = current_hour // 4
        days_elapsed = int(self.current_time / 86400)
        day_of_week = days_elapsed % 7
        day_type = 'weekend' if day_of_week >= 5 else 'weekday'
        key = (time_slot, day_type, severity, incident_h3)

        # 1. 実績ベースの事前計算モデルから検索
        hospital_probs = self.hospital_selection_model.get(key)

        if hospital_probs:
            if self.verbose_logging:
                print(f"[INFO] 事前計算モデル（実績ベース）を使用: {len(hospital_probs)}候補")
        else:
            # 2. ★変更点：ヒットしない場合、静的フォールバックモデルから検索
            if hasattr(self, 'static_fallback_model'):
                hospital_probs = self.static_fallback_model.get(severity, {}).get(incident_h3)
                if hospital_probs:
                     if self.verbose_logging:
                        print(f"[INFO] ★静的フォールバックモデル★を使用: {severity}, 候補数: {len(hospital_probs)}")
                else:
                    # 静的フォールバックにもない場合は、最終手段として決定論的選択
                    if self.verbose_logging:
                        print(f"[INFO] フォールバックモデルにも候補なし、決定論的選択にフォールバック")
                    return self._select_hospital_deterministic(incident_h3, severity)
            else:
                 # 完全にフォールバック
                if self.verbose_logging:
                    print(f"[INFO] 確率モデルなし、決定論的選択にフォールバック")
                return self._select_hospital_deterministic(incident_h3, severity)

        # 確率的選択の実行（変更なし）
        selected_hospital = self._probabilistic_selection(hospital_probs)
        
        # デバッグ：選択された病院までの距離を表示
        if selected_hospital and self.verbose_logging:
            try:
                inc_lat, inc_lon = h3.cell_to_latlng(incident_h3)
                hosp_info = self.hospital_data[self.hospital_data['h3_index'] == selected_hospital]
                if not hosp_info.empty:
                    hosp_lat = hosp_info.iloc[0]['latitude']
                    hosp_lon = hosp_info.iloc[0]['longitude']
                    distance = self._calculate_distance(inc_lat, inc_lon, hosp_lat, hosp_lon)
                    print(f"[DEBUG] 選択病院までの距離: {distance:.2f}km (傷病度: {severity})")
            except Exception as e:
                if self.verbose_logging:
                    print(f"[WARN] 距離計算でエラー: {e}")

        # 統計の更新（選択方法も記録）
        if selected_hospital:
            # 選択方法を判定
            if key in self.hospital_selection_model:
                selection_method = 'probabilistic_success'
            elif hasattr(self, 'static_fallback_model') and self.static_fallback_model.get(severity, {}).get(incident_h3):
                selection_method = 'static_fallback_used'
            else:
                selection_method = 'deterministic_fallback'
            
            self._update_selection_statistics(selected_hospital, severity, selection_method)

        return selected_hospital
    
    def _probabilistic_selection(self, hospital_probs: Dict[str, float]) -> str:
        """確率分布に基づいて病院を選択"""
        
        if not hospital_probs:
            return None
        
        # NumPyの確率的選択を使用
        hospitals = list(hospital_probs.keys())
        probabilities = list(hospital_probs.values())
        
        # デバッグ: 確率値の型をチェック
        if self.verbose_logging:
            print(f"[DEBUG] hospital_probs型チェック: {[(h, type(p), p) for h, p in hospital_probs.items()]}")
        
        # 確率値の型を修正（文字列が混入している場合の対処）
        try:
            probabilities = [float(p) for p in probabilities]
        except (ValueError, TypeError) as e:
            print(f"[ERROR] 確率値の変換エラー: {e}")
            print(f"[ERROR] 問題のある確率値: {[(h, type(p), p) for h, p in hospital_probs.items()]}")
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
    


    def _update_selection_statistics(self, hospital_h3: str, severity: str, method: str = 'unknown'):
        """選択統計の更新（確率的選択用）- 選択方法も記録"""
        
        # 病院種別の判定
        if hospital_h3 in self.tertiary_hospitals:
            selection_type = 'tertiary'
            self.statistics['hospital_selection_stats']['tertiary_selections'] += 1
        elif hospital_h3 in self.secondary_primary_hospitals:
            selection_type = 'secondary_primary'
            self.statistics['hospital_selection_stats']['secondary_primary_selections'] += 1
        else:
            selection_type = 'default'
        
        # 詳細統計の更新（選択方法も含む）
        self._update_hospital_selection_stats(severity, selection_type, method)
    
    def _find_nearest_hospital(self, incident_h3: str, hospital_candidates: set) -> str:
        """指定された病院候補群から最寄りの病院を検索"""
        if not hospital_candidates:
            return incident_h3
        
        min_time = float('inf')
        nearest_hospital = list(hospital_candidates)[0]
        
        for hospital_h3 in hospital_candidates:
            travel_time = self.get_travel_time(incident_h3, hospital_h3, phase='transport')
            if travel_time < min_time:
                min_time = travel_time
                nearest_hospital = hospital_h3
        
        return nearest_hospital
    
    def _update_hospital_selection_stats(self, severity: str, selection_type: str, method: str = 'unknown'):
        """統計情報の詳細化 - 選択方法も記録"""
        if severity not in self.statistics['hospital_selection_stats']['by_severity']:
            self.statistics['hospital_selection_stats']['by_severity'][severity] = {
                'tertiary': 0,
                'secondary_primary': 0,
                'default': 0,
                'probabilistic_success': 0,
                'deterministic_fallback': 0,
                'static_fallback_used': 0,
                'error_fallback': 0
            }
        
        # 病院種別の統計更新
        if selection_type == 'tertiary':
            self.statistics['hospital_selection_stats']['by_severity'][severity]['tertiary'] += 1
        elif selection_type == 'secondary_primary':
            self.statistics['hospital_selection_stats']['by_severity'][severity]['secondary_primary'] += 1
        else:
            self.statistics['hospital_selection_stats']['by_severity'][severity]['default'] += 1
        
        # 選択方法の統計更新（安全にキーの存在を確認）
        if method in ['probabilistic_success', 'deterministic_fallback', 'static_fallback_used', 'error_fallback']:
            # 傷病度別統計
            if method not in self.statistics['hospital_selection_stats']['by_severity'][severity]:
                self.statistics['hospital_selection_stats']['by_severity'][severity][method] = 0
            self.statistics['hospital_selection_stats']['by_severity'][severity][method] += 1
            
            # 全体統計
            if 'selection_methods' not in self.statistics['hospital_selection_stats']:
                self.statistics['hospital_selection_stats']['selection_methods'] = {
                    'probabilistic_success': 0,
                    'deterministic_fallback': 0,
                    'static_fallback_used': 0,
                    'error_fallback': 0
                }
            self.statistics['hospital_selection_stats']['selection_methods'][method] += 1
    
    def process_event(self, event: Event):
        """イベントの処理"""
        old_time = self.current_time
        self.current_time = event.time
        
        # 待機時間の更新（休憩機能が有効な場合のみ）
        if self.enable_breaks:
            for ambulance in self.ambulances.values():
                if ambulance.status == AmbulanceStatus.AVAILABLE and not ambulance.is_on_break:
                    self._update_standby_time(ambulance, old_time, self.current_time)
        
        # 深夜帯の活動追跡
        hour = int((self.current_time / 3600) % 24)
        if hour >= 22 or hour < 6:  # 深夜帯
            for ambulance in self.ambulances.values():
                if ambulance.night_activity_start_time is None:
                    ambulance.night_activity_start_time = self.current_time
        else:  # 深夜帯終了
            for ambulance in self.ambulances.values():
                if hour == 6 and ambulance.night_activity_start_time is not None:
                    # 深夜帯終了時のリセット（休憩を取っていない場合）
                    if not ambulance.is_on_break:
                        ambulance.night_activity_duration = 0.0
                        ambulance.night_activity_count = 0
                        ambulance.night_activity_start_time = None
        
        if self.verbose_logging and event.event_type != EventType.NEW_CALL:
             print(f"[DEBUG] Processing event: {event.event_type} for data {event.data} at time {self.current_time:.2f} (delta: {self.current_time - old_time:.2f}s)")
        
        if event.event_type == EventType.NEW_CALL:
            if self.verbose_logging:
                 print(f"[INFO] Event: NEW_CALL for {event.data['call_id']} at H3 {event.data['h3_index']} (Severity: {event.data['severity']}) at time {self.current_time:.2f}")
            self._handle_new_call(event)
        elif event.event_type == EventType.ARRIVE_SCENE:
            self._handle_arrive_scene(event)
        elif event.event_type == EventType.DEPART_SCENE:
            self._handle_depart_scene(event)
        elif event.event_type == EventType.ARRIVE_HOSPITAL:
            self._handle_arrive_hospital(event)
        elif event.event_type == EventType.AMBULANCE_AVAILABLE:
            self._handle_ambulance_available(event)
        elif event.event_type == EventType.BREAK_END:
            self._handle_break_end(event)
    
    def _handle_new_call(self, event: Event):
        """新規事案の処理"""
        call = EmergencyCall(
            id=event.data['call_id'],
            time=event.time,
            h3_index=event.data['h3_index'],
            severity=event.data['severity'],
            call_datetime=event.data.get('call_datetime')
        )
        self.calls[call.id] = call
        
        self.statistics['total_calls'] += 1
        hour = int(event.time // 3600) % 24
        self.statistics['calls_by_hour'][hour] += 1
        
        # 傷病度を渡すように修正
        ambulance = self.find_closest_available_ambulance(call.h3_index, call.severity)
        
        if ambulance:
            ambulance.status = AmbulanceStatus.DISPATCHED
            ambulance.assigned_call = call.id
            call.assigned_ambulance = ambulance.id
            call.dispatch_time = event.time
            
            # 深夜帯の活動カウント
            if hour >= 22 or hour < 6:
                ambulance.night_activity_count += 1
            
            if self.verbose_logging:
                print(f"[ACTION] Call {call.id}: Ambulance {ambulance.id} dispatched from {ambulance.current_h3_index} at {event.time:.2f}. Status: {ambulance.status}")

            travel_time_to_scene = self.get_travel_time_for_actual_movement(
                ambulance.current_h3_index, call.h3_index, phase='response', 
                context='dispatch_to_scene')
            
            arrive_event = Event(
                time=event.time + travel_time_to_scene,
                event_type=EventType.ARRIVE_SCENE,
                data={'call_id': call.id, 'ambulance_id': ambulance.id}
            )
            heapq.heappush(self.event_queue, arrive_event)
            if self.verbose_logging:
                print(f"[SCHEDULE] Call {call.id}: ARRIVE_SCENE event for Amb {ambulance.id} scheduled at {arrive_event.time:.2f} (travel: {travel_time_to_scene:.2f}s)")
        else:
            if self.verbose_logging:
                print(f"[WARN] Call {call.id}: No ambulance available for dispatch at {event.time:.2f}. Call queued implicitly.")
    
    def _handle_arrive_scene(self, event: Event):
        """現場到着イベントの処理"""
        call_id = event.data['call_id']
        ambulance_id = event.data['ambulance_id']
        call = self.calls[call_id]
        ambulance = self.ambulances[ambulance_id]
        
        # 移動距離を記録
        distance = self.get_travel_distance(ambulance.station_h3_index, call.h3_index, 'dispatch_to_scene')
        call.dispatch_to_scene_distance = distance
        self.statistics['travel_distances']['dispatch_to_scene'].append(distance)
        self.statistics['total_distance'] += distance
        
        if call.severity not in self.statistics['travel_distances_by_severity']:
            self.statistics['travel_distances_by_severity'][call.severity] = {
                'dispatch_to_scene': [],
                'scene_to_hospital': [],
                'hospital_to_station': []
            }
        self.statistics['travel_distances_by_severity'][call.severity]['dispatch_to_scene'].append(distance)
        
        if self.verbose_logging:
            print(f"[DISTANCE] Call {call.id}: Dispatch to scene distance recorded: {distance:.1f}km")

        ambulance.status = AmbulanceStatus.ON_SCENE
        ambulance.current_h3_index = call.h3_index
        call.arrive_scene_time = event.time
        
        if self.verbose_logging:
            print(f"[ACTION] Call {call.id}: Amb {ambulance.id} arrived scene at {call.h3_index} at {event.time:.2f}. Status: {ambulance.status}")

        if call.dispatch_time is not None:
            dispatch_to_scene_duration = (event.time - call.dispatch_time)
            self.statistics['phase_durations']['dispatch_to_scene'].append(dispatch_to_scene_duration / 60.0)
            if call.severity not in self.statistics['phase_durations_by_severity']:
                self.statistics['phase_durations_by_severity'][call.severity] = {k: [] for k in self.statistics['phase_durations'].keys()}
            self.statistics['phase_durations_by_severity'][call.severity]['dispatch_to_scene'].append(dispatch_to_scene_duration / 60.0)
            
            call.response_time = dispatch_to_scene_duration / 60.0
            self.statistics['response_times'].append(call.response_time)
            severity = call.severity
            if severity not in self.statistics['response_times_by_severity']:
                self.statistics['response_times_by_severity'][severity] = []
            self.statistics['response_times_by_severity'][severity].append(call.response_time)

            # 閾値チェック
            if call.response_time <= 6.0: 
                self.statistics['threshold_6min']['achieved'] += 1
            self.statistics['threshold_6min']['total'] += 1
            if call.response_time <= 13.0: 
                self.statistics['threshold_13min']['achieved'] += 1
            self.statistics['threshold_13min']['total'] += 1

            for th_key, th_val in [('threshold_6min_by_severity', 6.0), ('threshold_13min_by_severity', 13.0)]:
                if severity not in self.statistics[th_key]: 
                    self.statistics[th_key][severity] = {'achieved': 0, 'total': 0}
                if call.response_time <= th_val: 
                    self.statistics[th_key][severity]['achieved'] += 1
                self.statistics[th_key][severity]['total'] += 1

        # 現場滞在時間を傷病度別に生成
        if hasattr(self.service_time_generator, 'generate_time'):
            # generate_timeメソッドの引数を確認
            import inspect
            sig = inspect.signature(self.service_time_generator.generate_time)
            if 'call_datetime' in sig.parameters:
                # 拡張版の場合
                on_scene_time_seconds = self.service_time_generator.generate_time(
                    call.severity, 'on_scene_time', call_datetime=call.call_datetime
                ) * 60
            else:
                # 従来版の場合
                on_scene_time_seconds = self.service_time_generator.generate_time(
                    call.severity, 'on_scene_time'
                ) * 60
        
        depart_event = Event(
            time=event.time + on_scene_time_seconds,
            event_type=EventType.DEPART_SCENE,
            data={'call_id': call.id, 'ambulance_id': ambulance.id}
        )
        heapq.heappush(self.event_queue, depart_event)
        if self.verbose_logging:
            print(f"[SCHEDULE] Call {call.id}: DEPART_SCENE event for Amb {ambulance.id} scheduled at {depart_event.time:.2f} (on-scene time: {on_scene_time_seconds:.2f}s)")
    
    def _handle_depart_scene(self, event: Event):
        """現場出発の処理"""
        call = self.calls.get(event.data['call_id'])
        ambulance = self.ambulances.get(event.data['ambulance_id'])

        if not call or not ambulance:
            if self.verbose_logging: 
                print(f"[ERROR] DEPART_SCENE: Call or Ambulance not found. Data: {event.data}")
            return
        
        call.depart_scene_time = event.time
        if call.arrive_scene_time is not None:
            on_scene_actual_duration = (event.time - call.arrive_scene_time)
            call.on_scene_duration = on_scene_actual_duration / 60.0
            self.statistics['phase_durations']['on_scene'].append(call.on_scene_duration)

        # ★★★ここからが大きな変更点★★★
        hospital_h3 = self.select_hospital(call.h3_index, call.severity)

        if hospital_h3 is None:
            # 搬送先が見つからなかった場合
            if self.verbose_logging:
                print(f"[ACTION] Call {call.id}: No hospital found. Amb {ambulance.id} will return to station from scene.")
            
            # 現場から直接帰署する
            ambulance.status = AmbulanceStatus.RETURNING
            return_time_seconds = self.get_travel_time_for_actual_movement(
                ambulance.current_h3_index, ambulance.station_h3_index, phase='return',
                context='scene_to_station')
            available_event = Event(
                time=event.time + return_time_seconds,
                event_type=EventType.AMBULANCE_AVAILABLE,
                data={'call_id': call.id, 'ambulance_id': ambulance.id}
            )
            heapq.heappush(self.event_queue, available_event)
        else:
            # 搬送先が見つかった場合（元のロジック）
            if self.verbose_logging:
                    print(f"[INFO] Call {call.id}: Amb {ambulance.id} departing scene at {event.time:.2f}. Selected hospital H3: {hospital_h3} for severity: {call.severity}")

            transport_time_seconds = self.get_travel_time_for_actual_movement(
                call.h3_index, hospital_h3, phase='transport', 
                context='scene_to_hospital')
            
            ambulance.status = AmbulanceStatus.TRANSPORTING
            if self.verbose_logging:
                print(f"[ACTION] Call {call.id}: Amb {ambulance.id} status changed to TRANSPORTING.")
            
            arrive_hospital_event = Event(
                time=event.time + transport_time_seconds,
                event_type=EventType.ARRIVE_HOSPITAL,
                data={'call_id': call.id, 'ambulance_id': ambulance.id, 'hospital_h3': hospital_h3}
            )
            heapq.heappush(self.event_queue, arrive_hospital_event)
            if self.verbose_logging:
                print(f"[SCHEDULE] Call {call.id}: ARRIVE_HOSPITAL event for Amb {ambulance.id} at H3 {hospital_h3} scheduled at {arrive_hospital_event.time:.2f} (transport time: {transport_time_seconds:.2f}s)")
   
    def _handle_arrive_hospital(self, event: Event):
       """病院到着イベントの処理"""
       call_id = event.data['call_id']
       ambulance_id = event.data['ambulance_id']
       call = self.calls[call_id]
       ambulance = self.ambulances[ambulance_id]
       hospital_h3 = event.data['hospital_h3']

       if not call or not ambulance:
           if self.verbose_logging: 
               print(f"[ERROR] ARRIVE_HOSPITAL: Call or Ambulance not found. Data: {event.data}")
           return

       ambulance.status = AmbulanceStatus.AT_HOSPITAL
       ambulance.current_h3_index = hospital_h3
       call.arrive_hospital_time = event.time
       
       if self.verbose_logging:
           print(f"[ACTION] Call {call.id}: Amb {ambulance.id} arrived hospital {hospital_h3} at {event.time:.2f}. Status: {ambulance.status}")

       if call.depart_scene_time is not None:
           transport_actual_duration = (event.time - call.depart_scene_time)
           call.transport_duration = transport_actual_duration / 60.0
           self.statistics['phase_durations']['scene_to_hospital'].append(call.transport_duration)
           if call.severity not in self.statistics['phase_durations_by_severity']:
               self.statistics['phase_durations_by_severity'][call.severity] = {k: [] for k in self.statistics['phase_durations'].keys()}
           self.statistics['phase_durations_by_severity'][call.severity]['scene_to_hospital'].append(call.transport_duration)

       # 病院滞在時間を傷病度別に生成
       if hasattr(self.service_time_generator, 'generate_time'):
           import inspect
           sig = inspect.signature(self.service_time_generator.generate_time)
           if 'call_datetime' in sig.parameters:
               # 拡張版の場合
               hospital_time_seconds = self.service_time_generator.generate_time(
                   call.severity, 'hospital_time', call_datetime=call.call_datetime
               ) * 60
           else:
               # 従来版の場合
               hospital_time_seconds = self.service_time_generator.generate_time(
                   call.severity, 'hospital_time'
               ) * 60
       
       call.depart_hospital_time = event.time + hospital_time_seconds
       
       return_time_seconds = self.get_travel_time_for_actual_movement(
           hospital_h3, ambulance.station_h3_index, phase='return', 
           context='hospital_to_station')
       
       available_event_time = call.depart_hospital_time + return_time_seconds
       
       available_event = Event(
           time=available_event_time,
           event_type=EventType.AMBULANCE_AVAILABLE,
           data={'call_id': call.id, 'ambulance_id': ambulance.id}
       )
       heapq.heappush(self.event_queue, available_event)
       if self.verbose_logging:
           print(f"[SCHEDULE] Call {call.id}: AMBULANCE_AVAILABLE event for Amb {ambulance.id} scheduled at {available_event.time:.2f} (hospital_time: {hospital_time_seconds:.2f}s, return_time: {return_time_seconds:.2f}s)")

       # 移動距離を記録
       distance = self.get_travel_distance(call.h3_index, hospital_h3, 'scene_to_hospital')
       call.scene_to_hospital_distance = distance
       self.statistics['travel_distances']['scene_to_hospital'].append(distance)
       self.statistics['total_distance'] += distance
       
       if call.severity not in self.statistics['travel_distances_by_severity']:
           self.statistics['travel_distances_by_severity'][call.severity] = {
               'dispatch_to_scene': [],
               'scene_to_hospital': [],
               'hospital_to_station': []
           }
       self.statistics['travel_distances_by_severity'][call.severity]['scene_to_hospital'].append(distance)
       
       if self.verbose_logging:
           print(f"[DISTANCE] Call {call.id}: Scene to hospital distance recorded: {distance:.1f}km")

    def _handle_ambulance_available(self, event: Event):
       """救急車が利用可能になったイベントの処理"""
       ambulance_id = event.data['ambulance_id']
       ambulance = self.ambulances[ambulance_id]

       # イベントデータからトリガーとなった事案IDを取得 (存在しない場合も考慮)
       triggering_call_id = event.data.get('call_id') 

       if not ambulance:
           if self.verbose_logging: 
               print(f"[ERROR] AMBULANCE_AVAILABLE: Ambulance {ambulance_id} not found. Data: {event.data}")
           return

       # ログ出力用に、トリガーIDがNoneの場合の表示を調整
       log_trigger_id = triggering_call_id if triggering_call_id else 'N/A (e.g., initial setup or direct event)'
       if self.verbose_logging:
           print(f"[INFO] AMBULANCE_AVAILABLE event for Amb {ambulance.id} (triggered by call: {log_trigger_id}) at time {event.time:.2f}. Current status: {ambulance.status}")

       # 既に完全に利用可能（AVAILABLEかつ未割り当て）な場合は、重複イベントとして処理をスキップ
       if ambulance.status == AmbulanceStatus.AVAILABLE and ambulance.assigned_call is None:
           if self.verbose_logging:
               print(f"[WARN] Amb {ambulance.id} is already AVAILABLE and not assigned. Skipping redundant AMBULANCE_AVAILABLE event.")
           return

       # このイベントが初期利用不能状態からの復帰によるものかを判定
       is_initial_recovery = False
       if triggering_call_id and triggering_call_id.startswith("initial_unavailable_"):
           is_initial_recovery = True
       
       # 深夜帯の活動時間更新
       hour = int((self.current_time / 3600) % 24)
       if (hour >= 22 or hour < 6) and ambulance.assigned_call and not is_initial_recovery:
           # 出場から帰署までの時間を加算
           if triggering_call_id:
               call = self.calls.get(triggering_call_id)
               if call and call.dispatch_time is not None:
                   activity_duration = event.time - call.dispatch_time
                   ambulance.night_activity_duration += activity_duration
       
       # --- 統計情報更新処理 (実際の事案完了時のみ行う) ---
       if not is_initial_recovery and triggering_call_id:
           # triggering_call_id があり、かつ初期復帰ではない場合、実際の事案完了とみなす
           call = self.calls.get(triggering_call_id)
           if call:
               # 病院から署への移動距離を記録
               if call.arrive_hospital_time is not None:
                   # 病院のH3インデックスを取得（ambulance.current_h3_indexは病院を示している）
                   hospital_h3 = ambulance.current_h3_index
                   distance = self.get_travel_distance(hospital_h3, ambulance.station_h3_index, 'hospital_to_station')
                   call.hospital_to_station_distance = distance
                   call.total_distance = (call.dispatch_to_scene_distance or 0) + (call.scene_to_hospital_distance or 0) + distance
                   self.statistics['travel_distances']['hospital_to_station'].append(distance)
                   self.statistics['total_distance'] += distance
                   
                   if call.severity not in self.statistics['travel_distances_by_severity']:
                       self.statistics['travel_distances_by_severity'][call.severity] = {
                           'dispatch_to_scene': [],
                           'scene_to_hospital': [],
                           'hospital_to_station': []
                       }
                   self.statistics['travel_distances_by_severity'][call.severity]['hospital_to_station'].append(distance)
                   
                   if self.verbose_logging:
                       print(f"[DISTANCE] Call {call.id}: Hospital to station distance recorded: {distance:.1f}km")
                       print(f"[DISTANCE] Call {call.id}: Total distance for this call: {call.total_distance:.1f}km")
               # 事案完了に伴う時刻記録
               call.return_to_station_time = event.time
               call.completion_time = event.time

               # 各フェーズの所要時間記録 (病院滞在、帰署)
               if call.arrive_hospital_time is not None and call.depart_hospital_time is not None:
                   actual_hospital_duration = (call.depart_hospital_time - call.arrive_hospital_time)
                   call.hospital_duration = actual_hospital_duration / 60.0
                   self.statistics['phase_durations']['at_hospital'].append(call.hospital_duration)
                   if call.severity not in self.statistics['phase_durations_by_severity']:
                        self.statistics['phase_durations_by_severity'][call.severity] = {k: [] for k in self.statistics['phase_durations'].keys()}
                   self.statistics['phase_durations_by_severity'][call.severity]['at_hospital'].append(call.hospital_duration)
               
               if call.depart_hospital_time is not None:
                   return_actual_duration = (event.time - call.depart_hospital_time)
                   call.return_duration = return_actual_duration / 60.0
                   self.statistics['phase_durations']['hospital_to_station'].append(call.return_duration)
                   if call.severity not in self.statistics['phase_durations_by_severity']:
                        self.statistics['phase_durations_by_severity'][call.severity] = {k: [] for k in self.statistics['phase_durations'].keys()}
                   self.statistics['phase_durations_by_severity'][call.severity]['hospital_to_station'].append(call.return_duration)

               # 全体統計の更新
               self.statistics['completed_calls'] += 1
               if call.dispatch_time is not None: # dispatch_time がないと活動時間は計算できない
                    ambulance_active_duration_for_this_call = event.time - call.dispatch_time
                    self.statistics['ambulance_utilization'][ambulance.id]['active_time'] += ambulance_active_duration_for_this_call
               
               # 救急車ごとの統計更新
               self.statistics['ambulance_utilization'][ambulance.id]['calls_handled'] += 1
               ambulance.num_calls_handled +=1 # Ambulanceオブジェクトのフィールドも更新
               
               if call.time is not None and call.completion_time is not None:
                   incident_total_activity_time = call.completion_time - call.time
                   self.statistics['total_activity_times'].append(incident_total_activity_time)
           else:
               # triggering_call_id があったが、該当する call オブジェクトが見つからなかった場合
               # (例: 事案データが何らかの理由で消去されたなど、通常は考えにくいケース)
               if self.verbose_logging:
                   print(f"[WARN] AMBULANCE_AVAILABLE: Triggering call {triggering_call_id} (expected to be a real call) not found for Amb {ambulance.id}. Statistics for this call might be incomplete. Ambulance will still become available.")
       elif is_initial_recovery:
           # 初期利用不能状態からの復帰の場合のログ (統計更新は行わない)
           if self.verbose_logging:
               print(f"[INFO] Amb {ambulance.id} is becoming available from initial unavailable state. No call-specific statistics updated for this event.")
       else: 
           # triggering_call_id が None または予期せぬ値の場合 (初期復帰でもなく、有効な事案IDでもない)
           # (例: 何らかの理由で直接 AMBULANCE_AVAILABLE イベントが投入された場合など)
           if self.verbose_logging:
               print(f"[WARN] AMBULANCE_AVAILABLE: Event for Amb {ambulance.id} has an unexpected or missing triggering_call_id ('{triggering_call_id}'). Ambulance will become available without call-specific statistics.")

       # --- 救急車の状態更新 (このイベント処理の主目的。上記統計処理の成否に関わらず必ず実行) ---
       original_status_before_available = ambulance.status # ログ用
       
       # 自署に帰還した場合のみ休憩判定（休憩機能が有効な場合）
       if self.enable_breaks and ambulance.current_h3_index == ambulance.station_h3_index and not is_initial_recovery:
           # 8:30付近での乗務員交代チェック
           if self._should_change_crew(ambulance, self.current_time):
               self._handle_crew_change(ambulance, self.current_time)
           else:
               # 休憩要件チェック（交代していない場合のみ）
               if not ambulance.is_on_break:
                   # レストタイムチェック
                   rest_reason = self._check_rest_time_requirements(ambulance, self.current_time)
                   if rest_reason:
                       self._start_break(ambulance, 'rest_time', 30, rest_reason)
                       return  # 休憩に入るため、AVAILABLEにはしない
                   
                   # 出場間インターバルチェック
                   elif self._check_interval_requirements(ambulance, self.current_time):
                       self._start_break(ambulance, 'interval', 60, 'night_interval')
                       return  # 休憩に入るため、AVAILABLEにはしない
       
       # 通常のAVAILABLE処理
       ambulance.status = AmbulanceStatus.AVAILABLE
       ambulance.current_h3_index = ambulance.station_h3_index # 署に戻す
       previous_assigned_call_id_on_ambulance = ambulance.assigned_call # ログ用
       ambulance.assigned_call = None # 次の事案に備えてクリア
       
       if self.verbose_logging:
           log_message = f"[ACTION] Amb {ambulance.id} is now AVAILABLE at station {ambulance.station_h3_index} at {event.time:.2f}. Prev status: {original_status_before_available}."
           # ログにコンテキスト情報を追加
           if not is_initial_recovery and triggering_call_id and self.calls.get(triggering_call_id): # callオブジェクトが存在する場合のみ
               log_message += f" (Completed call: {triggering_call_id}, Amb's prev assigned: {previous_assigned_call_id_on_ambulance})"
           elif is_initial_recovery:
               log_message += f" (Initial recovery, Amb's prev assigned: {previous_assigned_call_id_on_ambulance})"
           else: # triggering_call_id がない、またはcallオブジェクトが見つからない場合
               log_message += f" (Context: {log_trigger_id}, Amb's prev assigned: {previous_assigned_call_id_on_ambulance})"
           print(log_message)
   
    def _handle_break_end(self, event: Event):
        """休憩終了イベントの処理"""
        ambulance_id = event.data['ambulance_id']
        ambulance = self.ambulances.get(ambulance_id)
        
        if not ambulance:
            if self.verbose_logging:
                print(f"[ERROR] BREAK_END: Ambulance {ambulance_id} not found.")
            return
        
        self._end_break(ambulance)
   
    def calculate_utilization(self):
       """稼働率の計算"""
       if not self.ambulances:
           return

       current_hour = int(self.current_time // 3600) % 24
       active_count = 0
       total_count = len(self.ambulances)
       
       for ambulance in self.ambulances.values():
           if ambulance.status != AmbulanceStatus.AVAILABLE:
               active_count += 1
       
       utilization = active_count / total_count if total_count > 0 else 0
       
       hour_key = current_hour 
       if hour_key not in self.statistics['utilization_by_hour']:
            self.statistics['utilization_by_hour'][hour_key] = []
            
       self.statistics['utilization_by_hour'][hour_key].append(utilization)
   
    def run(self, end_time: float = 86400, verbose: bool = False):
       """シミュレーションの実行"""
       self.verbose_logging = verbose
       print(f"シミュレーション開始（{end_time/3600:.1f}時間）, Verbose Logging: {self.verbose_logging}")
       self.configured_end_time = end_time
       
       initial_event_count = len(self.event_queue)
       
       with tqdm(total=initial_event_count, desc="イベント処理") as pbar:
           processed_event_count = 0
           while self.event_queue and self.event_queue[0].time <= end_time:
               event = heapq.heappop(self.event_queue)
               pbar.desc = f"イベント処理 (Q: {len(self.event_queue)}, T: {event.time/3600:.1f}h)"
               self.process_event(event)
               processed_event_count += 1
               
               if int(event.time) % 300 == 0:
                   self.calculate_utilization()
               
               if processed_event_count > pbar.n:
                   pbar.update(processed_event_count - pbar.n)

           if processed_event_count > pbar.n:
                pbar.update(processed_event_count - pbar.n)
           pbar.set_postfix_str(f"完了 (残りQ: {len(self.event_queue)})")
      
       print("シミュレーション完了")
       return self.generate_report()
   
    def generate_report(self):
        """シミュレーション結果のレポートを生成"""
        report = {'summary': {}, 'details': {}}
        
        # NumPy型をPython標準型に変換するヘルパー関数
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # NumPy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # NumPy array
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj
        
        # --- 全体サマリー ---
        completed_calls_count = int(self.statistics['completed_calls'])
        total_distance_km = float(self.statistics['total_distance']) if self.statistics['total_distance'] else 0.0

        report['summary'] = {
            'total_calls': int(self.statistics['total_calls']),
            'completed_calls': completed_calls_count,
            'simulation_duration_hours': float(self.configured_end_time / 3600),
            'total_distance_km': total_distance_km,
            'average_distance_per_call_km': float(total_distance_km / completed_calls_count) if completed_calls_count > 0 else 0.0,
        }
        
        # --- 応答時間 ---
        rt_overall = self.statistics['response_times']
        report['response_times'] = {
            'overall': {
                'mean': float(np.mean(rt_overall)) if rt_overall else 0.0,
                'median': float(np.median(rt_overall)) if rt_overall else 0.0,
                'std': float(np.std(rt_overall)) if rt_overall else 0.0,
                'min': float(np.min(rt_overall)) if rt_overall else 0.0,
                'max': float(np.max(rt_overall)) if rt_overall else 0.0,
                'count': int(len(rt_overall))
            },
            'by_severity': {
                sev: {
                    'mean': float(np.mean(times)) if times else 0.0,
                    'median': float(np.median(times)) if times else 0.0,
                    'std': float(np.std(times)) if times else 0.0,
                    'count': int(len(times))
                }
                for sev, times in self.statistics['response_times_by_severity'].items()
            }
        }

        # --- 閾値達成率 ---
        th6_total = int(self.statistics['threshold_6min']['total'])
        th6_achieved = int(self.statistics['threshold_6min']['achieved'])
        th13_total = int(self.statistics['threshold_13min']['total'])
        th13_achieved = int(self.statistics['threshold_13min']['achieved'])

        report['threshold_performance'] = {
            '6_minutes': {
                'total': th6_total,
                'achieved': th6_achieved,
                'rate': float(th6_achieved / th6_total * 100) if th6_total > 0 else 0.0
            },
            '13_minutes': {
                'total': th13_total,
                'achieved': th13_achieved,
                'rate': float(th13_achieved / th13_total * 100) if th13_total > 0 else 0.0
            },
            'by_severity': {
                '6_minutes': {
                    sev: {
                        'total': int(data['total']),
                        'achieved': int(data['achieved']),
                        'rate': float(data['achieved'] / data['total'] * 100) if data['total'] > 0 else 0.0
                    }
                    for sev, data in self.statistics['threshold_6min_by_severity'].items()
                },
                '13_minutes': {
                     sev: {
                        'total': int(data['total']),
                        'achieved': int(data['achieved']),
                        'rate': float(data['achieved'] / data['total'] * 100) if data['total'] > 0 else 0.0
                    }
                    for sev, data in self.statistics['threshold_13min_by_severity'].items()
                }
            }
        }
        
        # --- 稼働率 ---
        report['utilization'] = {
            'by_hour': {
                h: {
                    'mean': float(np.mean(rates)) if rates else 0.0,
                    'median': float(np.median(rates)) if rates else 0.0,
                    'std': float(np.std(rates)) if rates else 0.0
                }
                for h, rates in self.statistics['utilization_by_hour'].items() if rates
            },
            'by_ambulance': {
                amb_id: {
                    'active_time_hours': float(data['active_time'] / 3600),
                    'calls_handled': int(data['calls_handled']),
                    'utilization_rate': float(data['active_time'] / self.configured_end_time * 100) if self.configured_end_time > 0 else 0.0
                }
                for amb_id, data in self.statistics['ambulance_utilization'].items()
            }
        }
        
        # --- フェーズ別所要時間 ---
        report['phase_durations_stats'] = {
            phase: {
                'mean': float(np.mean(durations)) if durations else 0.0,
                'median': float(np.median(durations)) if durations else 0.0,
                'std': float(np.std(durations)) if durations else 0.0,
                'count': int(len(durations))
            }
            for phase, durations in self.statistics['phase_durations'].items()
        }
        
        # --- 病院選択 ---
        hospital_selection_stats_data = self.statistics.get('hospital_selection_stats', {})
        report['hospital_selection_stats'] = {
            'overall': {
                'tertiary_selections': int(hospital_selection_stats_data.get('tertiary_selections', 0)),
                'secondary_primary_selections': int(hospital_selection_stats_data.get('secondary_primary_selections', 0)),
                'no_hospital_found': int(hospital_selection_stats_data.get('no_hospital_found', 0)),
            },
            'by_severity': {
                sev: {
                    'tertiary': int(data.get('tertiary', 0)),
                    'secondary_primary': int(data.get('secondary_primary', 0)),
                    'default': int(data.get('default', 0)),
                    'probabilistic_success': int(data.get('probabilistic_success', 0)),
                    'deterministic_fallback': int(data.get('deterministic_fallback', 0)),
                    'static_fallback_used': int(data.get('static_fallback_used', 0)),
                    'error_fallback': int(data.get('error_fallback', 0))
                }
                for sev, data in hospital_selection_stats_data.get('by_severity', {}).items()
            },
            'selection_methods': {
                method: int(count)
                for method, count in hospital_selection_stats_data.get('selection_methods', {}).items()
            }
        }

        # --- 移動距離統計 ---
        report['travel_distance_stats'] = {
            'overall': {
                phase: {
                    'mean': float(np.mean(distances)) if distances else 0.0,
                    'median': float(np.median(distances)) if distances else 0.0,
                    'std': float(np.std(distances)) if distances else 0.0,
                    'min': float(np.min(distances)) if distances else 0.0,
                    'max': float(np.max(distances)) if distances else 0.0,
                    'count': int(len(distances)),
                    'total_km': float(np.sum(distances)) if distances else 0.0
                }
                for phase, distances in self.statistics['travel_distances'].items()
            },
            'by_severity': {}
        }
        
        # 傷病度別距離統計
        for severity, severity_distances in self.statistics['travel_distances_by_severity'].items():
            report['travel_distance_stats']['by_severity'][severity] = {
                phase: {
                    'mean': float(np.mean(distances)) if distances else 0.0,
                    'median': float(np.median(distances)) if distances else 0.0,
                    'count': int(len(distances))
                }
                for phase, distances in severity_distances.items()
            }

        # --- 移動距離 (サマリーに追加) ---
        report['summary']['distance_stats'] = {
            phase: {
                'mean': float(np.mean(distances)) if distances else 0.0,
                'median': float(np.median(distances)) if distances else 0.0,
                'std': float(np.std(distances)) if distances else 0.0
            }
            for phase, distances in self.statistics['travel_distances'].items()
        }

        # 総距離統計をサマリーに追加
        all_phase_distances = []
        for distances in self.statistics['travel_distances'].values():
            all_phase_distances.extend(distances)
        
        if all_phase_distances:
            report['summary']['total_distance_stats'] = {
                'total_distance_km': float(sum(all_phase_distances)),
                'average_distance_per_trip_km': float(np.mean(all_phase_distances)),
                'total_trips': int(len(all_phase_distances))
            }

        # --- 移動時間詳細分析結果 ---
        if self.enable_travel_time_analysis and self.travel_time_analyzer:
            print("\n移動時間詳細分析を実行中...")
            self.travel_time_analyzer.save_detailed_logs()
            self.travel_time_analyzer.generate_analysis_report()
            
            # レポートに分析結果を追加
            report['travel_time_analysis'] = {
                'same_grid_counts_by_phase': {phase: int(count) for phase, count in self.travel_time_analyzer.same_grid_counts.items()},
                'total_travel_records': int(len(self.travel_time_analyzer.travel_time_logs)),
                'distance_category_stats': {
                    category: {
                        'count': int(len(times)),
                        'mean_time': float(np.mean(times)) if times else 0.0,
                        'median_time': float(np.median(times)) if times else 0.0,
                        'std_time': float(np.std(times)) if times else 0.0
                    }
                    for category, times in self.travel_time_analyzer.distance_categories.items()
                },
                'context_breakdown': self._get_context_breakdown()
            }

        # 休憩統計を追加
        if self.enable_breaks:
            report['break_statistics'] = {
                'rest_time': {
                    'total_count': int(self.statistics['rest_time_stats']['total_count']),
                    'by_reason': {
                        reason: int(count) 
                        for reason, count in self.statistics['rest_time_stats']['by_reason'].items()
                    },
                    'by_section': {
                        str(section): int(count)
                        for section, count in self.statistics['rest_time_stats']['by_section'].items()
                    },
                    'by_hour': {
                        str(hour): int(count)
                        for hour, count in self.statistics['rest_time_stats']['by_hour'].items()
                        if count > 0
                    }
                },
                'interval': {
                    'total_count': int(self.statistics['interval_stats']['total_count']),
                    'by_hour': {
                        str(hour): int(count)
                        for hour, count in self.statistics['interval_stats']['by_hour'].items()
                        if count > 0
                    },
                    'average_activity_before': float(
                        np.mean(self.statistics['interval_stats']['average_activity_before_interval'])
                    ) if self.statistics['interval_stats']['average_activity_before_interval'] else 0.0
                },
                'crew_changes': {
                    'total_count': int(self.statistics['crew_changes']['total_count']),
                    'by_hour': {
                        str(hour): int(count)
                        for hour, count in self.statistics['crew_changes']['by_hour'].items()
                        if count > 0
                    }
                }
            }

        # 最終的にレポート全体をPython標準型に変換
        report = convert_numpy_types(report)
        
        return report

    def _get_context_breakdown(self):
        """コンテキスト別統計を取得"""
        if not self.travel_time_analyzer or not self.travel_time_analyzer.travel_time_logs:
            return {}
        
        df = pd.DataFrame(self.travel_time_analyzer.travel_time_logs)
        context_breakdown = {}
        
        for context in df['context'].unique():
            context_df = df[df['context'] == context]
            context_breakdown[context] = {
                'count': int(len(context_df)),
                'mean_time': float(context_df['travel_time_minutes'].mean()),
                'median_time': float(context_df['travel_time_minutes'].median()),
                'phases': {phase: int(count) for phase, count in dict(context_df['phase'].value_counts()).items()}
            }
        
        return context_breakdown

def get_versioned_output_dir(base_dir: str, date_str: str, duration_hours: int) -> str:
    """
    バージョン管理された出力ディレクトリパスを生成する。
    例: data/tokyo/simulation_results/20190101_168h_v1
    """
    folder_name = f"{date_str.replace('-', '')}_{duration_hours}h"
    version = 1
    while True:
        versioned_folder_name = f"{folder_name}_v{version}"
        output_path = os.path.join(base_dir, versioned_folder_name)
        
        # フォルダが存在しない場合は新規作成
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            return output_path
        
        version += 1

def create_simulation_visualizations_enhanced(report: Dict, output_dir: str):
    """シミュレーション結果の可視化 - 病院選択統計を含む拡張版"""
    
    os.makedirs(output_dir, exist_ok=True)
    font_family = 'Meiryo'
    
    # 統一された傷病度定数をインポート
    from constants import SEVERITY_LEVELS, SEVERITY_COLORS
    
    # 傷病度の標準順序を定義
    SEVERITY_ORDER = SEVERITY_LEVELS
    
    def sort_severities(severities_list):
        """傷病度リストを標準順序でソートする"""
        sorted_severities = [s for s in SEVERITY_ORDER if s in severities_list]
        # 標準順序にない傷病度を末尾に追加
        for s in severities_list:
            if s not in sorted_severities:
                sorted_severities.append(s)
        return sorted_severities
    
    # 1. 応答時間の分布
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if 'response_times' in report and report['response_times']['overall']['count'] > 0:
        mean_rt = report['response_times']['overall']['mean']
        std_rt = report['response_times']['overall']['std']
        
        # ダミーデータ生成（実際の実装では生の応答時間データを使用すべき）
        dummy_data = np.random.lognormal(np.log(mean_rt if mean_rt > 0 else 1), 
                                     (std_rt / mean_rt if mean_rt > 0 else 0.5), 
                                     report['response_times']['overall'].get('count',1000))
        dummy_data = dummy_data[dummy_data > 0]
        if len(dummy_data) == 0: 
            dummy_data = [mean_rt]
        plot_data_rt = dummy_data

        plt.hist(plot_data_rt, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(mean_rt, color='red', linestyle='--', linewidth=2, label=f'平均: {mean_rt:.1f}分')
        plt.axvline(6, color='green', linestyle='--', linewidth=2, label='6分閾値')
        plt.axvline(13, color='orange', linestyle='--', linewidth=2, label='13分閾値')
        plt.xlabel('応答時間（分）', fontfamily=font_family)
        plt.ylabel('頻度', fontfamily=font_family)
        plt.title('応答時間の分布', fontfamily=font_family)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, '応答時間データなし', ha='center', va='center', fontfamily=font_family)

    plt.subplot(1, 2, 2)
    severity_colors = SEVERITY_COLORS
    available_severities_rt = list(report['response_times']['by_severity'].keys())
    severities_sorted_rt = sort_severities(available_severities_rt)
    
    if severities_sorted_rt:
        means_sorted_rt = [report['response_times']['by_severity'][s]['mean'] for s in severities_sorted_rt]
        colors_for_plot_rt = [severity_colors.get(s, '#808080') for s in severities_sorted_rt]
        plt.bar(severities_sorted_rt, means_sorted_rt, color=colors_for_plot_rt)
        plt.xlabel('傷病度', fontfamily=font_family)
        plt.ylabel('平均応答時間（分）', fontfamily=font_family)
        plt.title('傷病度別平均応答時間', fontfamily=font_family)
        plt.xticks(rotation=45, ha='right', fontfamily=font_family)
        plt.grid(True, alpha=0.3)
        for i, (s, m) in enumerate(zip(severities_sorted_rt, means_sorted_rt)):
            plt.text(i, m + 0.2, f'{m:.1f}', ha='center', fontfamily=font_family)
    else:
        plt.text(0.5, 0.5, '傷病度別応答時間データなし', ha='center', va='center', fontfamily=font_family)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'response_time_distribution.png'), dpi=300)
    plt.close()
    
    # 2. 時間帯別の稼働率
    plt.figure(figsize=(12, 6))
    hours = list(range(24))
    utilization_means = [report['utilization']['by_hour'].get(h, {}).get('mean', 0) * 100 for h in hours]
    plt.bar(hours, utilization_means, alpha=0.7, color='steelblue')
    plt.xlabel('時間帯', fontfamily=font_family)
    plt.ylabel('平均稼働率（%）', fontfamily=font_family)
    plt.title('時間帯別救急隊稼働率', fontfamily=font_family)
    plt.xticks(hours, fontfamily=font_family)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hourly_utilization.png'), dpi=300)
    plt.close()
    
    # 3. 閾値達成率 (全体)
    plt.figure(figsize=(10, 6))
    thresholds_overall = ['全体: 6分以内', '全体: 13分以内']
    rates_overall = [report['threshold_performance']['6_minutes']['rate'], report['threshold_performance']['13_minutes']['rate']]
    bars_overall = plt.bar(thresholds_overall, rates_overall, color=['#4CAF50', '#FF9800'])
    plt.ylabel('達成率（%）', fontfamily=font_family)
    plt.title('応答時間閾値達成率 (全体)', fontfamily=font_family)
    plt.ylim(0, 100)
    for bar, rate in zip(bars_overall, rates_overall): 
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate:.1f}%', ha='center', fontfamily=font_family)
    plt.axhline(y=90, color='red', linestyle='--', label='目標: 90%')
    plt.legend(prop={'family': font_family})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_achievement_overall.png'), dpi=300)
    plt.close()

    # 3b. 閾値達成率 (傷病度別)
    if 'by_severity' in report['threshold_performance'] and \
       report['threshold_performance']['by_severity']['6_minutes'] and \
       report['threshold_performance']['by_severity']['13_minutes']:
        available_severities_6min = set(report['threshold_performance']['by_severity']['6_minutes'].keys())
        available_severities_13min = set(report['threshold_performance']['by_severity']['13_minutes'].keys())
        common_severities = list(available_severities_6min.intersection(available_severities_13min))
        severities = sort_severities(common_severities)

        if severities:
            rates_6min_severity = [report['threshold_performance']['by_severity']['6_minutes'].get(s, {}).get('rate', 0) for s in severities]
            rates_13min_severity = [report['threshold_performance']['by_severity']['13_minutes'].get(s, {}).get('rate', 0) for s in severities]
            x = np.arange(len(severities))
            width = 0.35
            fig, ax = plt.subplots(figsize=(12, 7))
            rects1 = ax.bar(x - width/2, rates_6min_severity, width, label='6分以内達成率', color='#4CAF50')
            rects2 = ax.bar(x + width/2, rates_13min_severity, width, label='13分以内達成率', color='#FF9800')
            ax.set_ylabel('達成率（%）', fontfamily=font_family)
            ax.set_title('応答時間閾値達成率 (傷病度別)', fontfamily=font_family)
            ax.set_xticks(x)
            ax.set_xticklabels(severities, rotation=45, ha='right', fontfamily=font_family)
            ax.legend(prop={'family': font_family})
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, 100)
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontfamily=font_family)
            autolabel(rects1); autolabel(rects2)
            fig.tight_layout()
            plt.savefig(os.path.join(output_dir, 'threshold_achievement_by_severity.png'), dpi=300)
            plt.close(fig)

    # 4. 救急車別稼働率（上位20台）
    plt.figure(figsize=(15, 8))
    amb_data = report['utilization']['by_ambulance']
    if amb_data:
        sorted_ambs = sorted(amb_data.items(), key=lambda x_item: x_item[1]['utilization_rate'], reverse=True)[:20]
        amb_names = [amb[0] for amb in sorted_ambs]
        utilizations_amb = [amb[1]['utilization_rate'] for amb in sorted_ambs]
        call_counts_amb = [amb[1]['calls_handled'] for amb in sorted_ambs]
        x_amb = np.arange(len(amb_names))
        width_amb = 0.35
        fig, ax1 = plt.subplots(figsize=(15, 8))
        bars1 = ax1.bar(x_amb - width_amb/2, utilizations_amb, width_amb, label='稼働率', color='steelblue')
        ax1.set_xlabel('救急隊', fontfamily=font_family)
        ax1.set_ylabel('稼働率（%）', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_xticks(x_amb)
        ax1.set_xticklabels(amb_names, rotation=45, ha='right', fontfamily=font_family)
        
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x_amb + width_amb/2, call_counts_amb, width_amb, label='出動回数', color='orange')
        ax2.set_ylabel('出動回数', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        ax1.set_title('救急隊別稼働率と出動回数（上位20隊）', fontfamily=font_family)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', prop={'family': font_family})
        ax2.legend(loc='upper right', prop={'family': font_family})
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ambulance_utilization.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 5. フェーズ別所要時間の可視化
    phase_stats = report.get('phase_durations_stats', {})
    if phase_stats:
        phase_names_mapping = {
            'dispatch_to_scene': '出場～現着',
            'on_scene': '現場活動',
            'scene_to_hospital': '現発～病着',
            'at_hospital': '病院滞在',
            'hospital_to_station': '帰署時間'
        }
        phase_names = [phase_names_mapping.get(p, p) for p in phase_stats.keys()]
        mean_durations = [phase_stats[p].get('mean', 0) for p in phase_stats.keys()]
        
        plt.figure(figsize=(12, 7))
        bars = plt.bar(phase_names, mean_durations, color='skyblue')
        plt.xlabel('活動フェーズ', fontfamily=font_family)
        plt.ylabel('平均所要時間（分）', fontfamily=font_family)
        plt.title('活動フェーズ別平均所要時間（全体）', fontfamily=font_family)
        plt.xticks(rotation=45, ha='right', fontfamily=font_family)
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        
        for bar_item, duration_item in zip(bars, mean_durations):
            plt.text(bar_item.get_x() + bar_item.get_width()/2, bar_item.get_height() + 0.5, 
                     f'{duration_item:.1f}分', ha='center', va='bottom', fontfamily=font_family)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'phase_durations_overall.png'), dpi=300)
        plt.close()

    # 6. 病院選択統計の可視化
    if 'hospital_selection_stats' in report:
        hospital_stats = report['hospital_selection_stats']
        
        # 6a. 全体の病院選択分布
        plt.figure(figsize=(10, 6))
        selection_types = ['3次救急', '2次以下']
        selection_counts = [
            hospital_stats['overall']['tertiary_selections'],
            hospital_stats['overall']['secondary_primary_selections']
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        bars = plt.bar(selection_types, selection_counts, color=colors)
        plt.ylabel('選択回数', fontfamily=font_family)
        plt.title('病院選択分布（全体）', fontfamily=font_family)
        plt.grid(True, alpha=0.3)
        
        # 数値ラベル追加
        for bar, count in zip(bars, selection_counts):
            percentage = count / sum(selection_counts) * 100 if sum(selection_counts) > 0 else 0
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(selection_counts)*0.01,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontfamily=font_family)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hospital_selection_overall.png'), dpi=300)
        plt.close()
        
        # 6b. 傷病度別病院選択
        severity_data = hospital_stats['by_severity']
        severities = sort_severities(list(severity_data.keys()))
        
        if severities:
            tertiary_counts = [severity_data[s]['tertiary'] for s in severities]
            secondary_counts = [severity_data[s]['secondary_primary'] for s in severities]
            default_counts = [severity_data[s]['default'] for s in severities]
            
            x = np.arange(len(severities))
            width = 0.25
            
            fig, ax = plt.subplots(figsize=(14, 8))
            bars1 = ax.bar(x - width, tertiary_counts, width, label='3次救急', color='#FF6B6B')
            bars2 = ax.bar(x, secondary_counts, width, label='2次以下', color='#4ECDC4')
            bars3 = ax.bar(x + width, default_counts, width, label='デフォルト', color='#95E1D3')
            
            ax.set_ylabel('選択回数', fontfamily=font_family)
            ax.set_title('傷病度別病院選択分布', fontfamily=font_family)
            ax.set_xticks(x)
            ax.set_xticklabels(severities, fontfamily=font_family)
            ax.legend(prop={'family': font_family})
            ax.grid(True, alpha=0.3)
            
            # 数値ラベル追加
            def add_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{int(height)}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom',
                                  fontfamily=font_family, fontsize=9)
            
            add_labels(bars1)
            add_labels(bars2)
            add_labels(bars3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'hospital_selection_by_severity.png'), dpi=300)
            plt.close()
        
        # 6c. 傷病度別病院選択率（積み上げ棒グラフ）
        if severities:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 各傷病度の合計を計算して割合を求める
            total_by_severity = [tertiary_counts[i] + secondary_counts[i] + default_counts[i] for i in range(len(severities))]
            
            tertiary_pct = [tertiary_counts[i] / total_by_severity[i] * 100 if total_by_severity[i] > 0 else 0 for i in range(len(severities))]
            secondary_pct = [secondary_counts[i] / total_by_severity[i] * 100 if total_by_severity[i] > 0 else 0 for i in range(len(severities))]
            default_pct = [default_counts[i] / total_by_severity[i] * 100 if total_by_severity[i] > 0 else 0 for i in range(len(severities))]
            
            bars1 = ax.bar(severities, tertiary_pct, label='3次救急', color='#FF6B6B')
            bars2 = ax.bar(severities, secondary_pct, bottom=tertiary_pct, label='2次以下', color='#4ECDC4')
            bars3 = ax.bar(severities, default_pct, bottom=[tertiary_pct[i] + secondary_pct[i] for i in range(len(severities))], label='デフォルト', color='#95E1D3')
            
            ax.set_ylabel('選択率（%）', fontfamily=font_family)
            ax.set_title('傷病度別病院選択率', fontfamily=font_family)
            ax.set_xticklabels(severities, fontfamily=font_family)
            ax.legend(prop={'family': font_family})
            ax.set_ylim(0, 100)
            
            # 各傷病度の総数をx軸ラベルに追加
            new_labels = [f'{s}\n(n={total_by_severity[i]})' for i, s in enumerate(severities)]
            ax.set_xticklabels(new_labels, fontfamily=font_family)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'hospital_selection_rates_by_severity.png'), dpi=300)
            plt.close()
            
        # 6d. 選択方法別統計
        if 'selection_methods' in hospital_stats:
            plt.figure(figsize=(12, 6))
            
            method_stats = hospital_stats['selection_methods']
            methods = list(method_stats.keys())
            counts = list(method_stats.values())
            
            method_labels = {
                'probabilistic_success': '確率的選択成功',
                'deterministic_fallback': '決定論的フォールバック',
                'static_fallback_used': '静的フォールバック使用',
                'error_fallback': 'エラー時フォールバック'
            }
            
            display_methods = [method_labels.get(m, m) for m in methods]
            colors = ['#2E8B57', '#FF6347', '#4682B4', '#DAA520']
            
            bars = plt.bar(display_methods, counts, color=colors[:len(methods)])
            plt.ylabel('使用回数', fontfamily=font_family)
            plt.title('病院選択方法別統計', fontfamily=font_family)
            plt.xticks(rotation=45, ha='right', fontfamily=font_family)
            
            # 数値とパーセンテージを表示
            total = sum(counts) if sum(counts) > 0 else 1
            for bar, count in zip(bars, counts):
                percentage = count / total * 100
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontfamily=font_family)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'hospital_selection_methods.png'), dpi=300)
            plt.close()

    # 移動距離の可視化を追加
    if 'travel_distance_stats' in report and report['travel_distance_stats']['overall']:
        plt.figure(figsize=(15, 10))
        
        # 全体統計
        plt.subplot(2, 1, 1)
        distance_stats = report['travel_distance_stats']['overall']
        phases = ['出動→現場', '現場→病院', '病院→消防署']
        phases_en = ['dispatch_to_scene', 'scene_to_hospital', 'hospital_to_station']
        
        means = [distance_stats.get(phase, {}).get('mean', 0) for phase in phases_en]
        medians = [distance_stats.get(phase, {}).get('median', 0) for phase in phases_en]
        counts = [distance_stats.get(phase, {}).get('count', 0) for phase in phases_en]
        
        x = np.arange(len(phases))
        width = 0.35
        
        rects1 = plt.bar(x - width/2, means, width, label='平均距離', color='steelblue')
        rects2 = plt.bar(x + width/2, medians, width, label='中央値距離', color='lightcoral')
        
        plt.xlabel('移動フェーズ', fontfamily=font_family)
        plt.ylabel('距離（キロメートル）', fontfamily=font_family)
        plt.title('移動フェーズ別の平均・中央値距離', fontfamily=font_family)
        plt.xticks(x, phases, fontfamily=font_family)
        plt.legend(prop={'family': font_family})
        plt.grid(True, alpha=0.3)
        
        # データラベルを追加
        for i, (rect1, rect2, count) in enumerate(zip(rects1, rects2, counts)):
            height1 = rect1.get_height()
            height2 = rect2.get_height()
            plt.annotate(f'{height1:.1f}km', xy=(rect1.get_x() + rect1.get_width() / 2, height1),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontfamily=font_family)
            plt.annotate(f'{height2:.1f}km', xy=(rect2.get_x() + rect2.get_width() / 2, height2),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontfamily=font_family)
            plt.text(i, -max(means+medians)*0.1, f'n={count}', ha='center', va='top', fontfamily=font_family)
        
        # 距離分布のヒストグラム
        plt.subplot(2, 1, 2)
        if 'summary' in report and 'distance_stats' in report['summary']:
            for i, (phase, phase_en) in enumerate(zip(phases, phases_en)):
                if phase_en in report['summary']['distance_stats']:
                    stats = report['summary']['distance_stats'][phase_en]
                    if stats['mean'] > 0:
                        # 仮想的な分布データを生成（実際のデータが利用できない場合）
                        mu = np.log(stats['mean']) if stats['mean'] > 0 else 0
                        sigma = stats['std'] / stats['mean'] if stats['mean'] > 0 and stats['std'] > 0 else 0.5
                        sample_data = np.random.lognormal(mu, sigma, 1000)
                        sample_data = sample_data[sample_data > 0]
                        plt.hist(sample_data, bins=30, alpha=0.6, label=phase, density=True)
            
            plt.xlabel('距離（キロメートル）', fontfamily=font_family)
            plt.ylabel('密度', fontfamily=font_family)
            plt.title('移動フェーズ別距離分布', fontfamily=font_family)
            plt.legend(prop={'family': font_family})
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'travel_distances.png'), dpi=300)
        plt.close()
        
        # 傷病度別距離統計
        if 'by_severity' in report['travel_distance_stats'] and report['travel_distance_stats']['by_severity']:
            plt.figure(figsize=(14, 8))
            severity_distance_stats = report['travel_distance_stats']['by_severity']
            severities = sort_severities(list(severity_distance_stats.keys()))
            
            if severities:
                phases_data = {phase: [] for phase in phases_en}
                
                for severity in severities:
                    if severity in severity_distance_stats:
                        for phase in phases_en:
                            mean_dist = severity_distance_stats[severity].get(phase, {}).get('mean', 0)
                            phases_data[phase].append(mean_dist)
                
                x = np.arange(len(severities))
                width = 0.25
                
                bars1 = plt.bar(x - width, phases_data['dispatch_to_scene'], width, label='出動→現場', alpha=0.8)
                bars2 = plt.bar(x, phases_data['scene_to_hospital'], width, label='現場→病院', alpha=0.8)
                bars3 = plt.bar(x + width, phases_data['hospital_to_station'], width, label='病院→署', alpha=0.8)
                
                # 数値ラベルを追加
                for i, (val1, val2, val3) in enumerate(zip(phases_data['dispatch_to_scene'], phases_data['scene_to_hospital'], phases_data['hospital_to_station'])):
                    plt.text(i - width, val1 + 0.1, f'{val1:.1f}', ha='center', va='bottom', fontfamily=font_family)
                    plt.text(i, val2 + 0.1, f'{val2:.1f}', ha='center', va='bottom', fontfamily=font_family)
                    plt.text(i + width, val3 + 0.1, f'{val3:.1f}', ha='center', va='bottom', fontfamily=font_family)
                
                plt.xlabel('傷病度', fontfamily=font_family)
                plt.ylabel('平均距離（キロメートル）', fontfamily=font_family)
                plt.title('傷病度別・移動フェーズ別平均距離', fontfamily=font_family)
                plt.xticks(x, severities, rotation=45, ha='right', fontfamily=font_family)
                plt.legend(prop={'family': font_family})
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'travel_distances_by_severity.png'), dpi=300)
                plt.close()

    # 休憩統計の可視化を追加
    if 'break_statistics' in report:
        break_stats = report['break_statistics']
        
        # 休憩取得状況の可視化
        plt.figure(figsize=(15, 10))
        
        # レストタイム取得理由別
        plt.subplot(2, 3, 1)
        if break_stats['rest_time']['by_reason']:
            reasons = list(break_stats['rest_time']['by_reason'].keys())
            counts = list(break_stats['rest_time']['by_reason'].values())
            reason_labels = {
                'lunch_period': '昼食(11-13時)',
                'lunch_extended': '昼食延長(15時)',
                'dinner_period': '夕食(17-19時)',
                'dinner_extended': '夕食延長(20時)'
            }
            display_reasons = [reason_labels.get(r, r) for r in reasons]
            plt.bar(display_reasons, counts, color='lightblue')
            plt.xlabel('取得理由', fontfamily=font_family)
            plt.ylabel('取得回数', fontfamily=font_family)
            plt.title('レストタイム取得理由別分布', fontfamily=font_family)
            plt.xticks(rotation=45, ha='right', fontfamily=font_family)
            for i, count in enumerate(counts):
                plt.text(i, count + 0.5, str(count), ha='center', fontfamily=font_family)
        
        # 方面別レストタイム取得数
        plt.subplot(2, 3, 2)
        if break_stats['rest_time']['by_section']:
            sections = sorted([int(s) for s in break_stats['rest_time']['by_section'].keys()])
            counts = [break_stats['rest_time']['by_section'][str(s)] for s in sections]
            plt.bar([f'方面{s}' for s in sections], counts, color='lightcoral')
            plt.xlabel('方面', fontfamily=font_family)
            plt.ylabel('レストタイム取得回数', fontfamily=font_family)
            plt.title('方面別レストタイム取得状況', fontfamily=font_family)
            plt.xticks(rotation=45, ha='right', fontfamily=font_family)
        
        # 時間帯別休憩取得状況
        plt.subplot(2, 3, 3)
        hours = list(range(24))
        rest_time_by_hour = [break_stats['rest_time']['by_hour'].get(str(h), 0) for h in hours]
        interval_by_hour = [break_stats['interval']['by_hour'].get(str(h), 0) for h in hours]
        
        x = np.arange(len(hours))
        width = 0.35
        plt.bar(x - width/2, rest_time_by_hour, width, label='レストタイム', color='skyblue')
        plt.bar(x + width/2, interval_by_hour, width, label='インターバル', color='orange')
        plt.xlabel('時間帯', fontfamily=font_family)
        plt.ylabel('取得回数', fontfamily=font_family)
        plt.title('時間帯別休憩取得状況', fontfamily=font_family)
        plt.xticks(x, hours, fontfamily=font_family)
        plt.legend(prop={'family': font_family})
        
        # 総休憩取得数
        plt.subplot(2, 3, 4)
        total_rest = break_stats['rest_time']['total_count']
        total_interval = break_stats['interval']['total_count']
        plt.bar(['レストタイム', 'インターバル'], [total_rest, total_interval], 
                color=['skyblue', 'orange'])
        plt.ylabel('総取得回数', fontfamily=font_family)
        plt.title('休憩種別総取得数', fontfamily=font_family)
        for i, (label, count) in enumerate([('レストタイム', total_rest), ('インターバル', total_interval)]):
            plt.text(i, count + 0.5, str(count), ha='center', fontfamily=font_family)
        
        # インターバル前の平均活動時間
        plt.subplot(2, 3, 5)
        avg_activity = break_stats['interval']['average_activity_before']
        if avg_activity > 0:
            plt.bar(['平均活動時間'], [avg_activity], color='darkblue')
            plt.ylabel('時間', fontfamily=font_family)
            plt.title('インターバル取得前の平均活動時間', fontfamily=font_family)
            plt.text(0, avg_activity + 0.1, f'{avg_activity:.1f}時間', ha='center', fontfamily=font_family)
        
        # 乗務員交代
        plt.subplot(2, 3, 6)
        crew_changes = break_stats['crew_changes']['total_count']
        plt.bar(['乗務員交代'], [crew_changes], color='green')
        plt.ylabel('交代回数', fontfamily=font_family)
        plt.title('総乗務員交代回数', fontfamily=font_family)
        plt.text(0, crew_changes + 0.5, str(crew_changes), ha='center', fontfamily=font_family)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'break_statistics.png'), dpi=300)
        plt.close()

# 実行スクリプト
def run_validation_simulation(
    target_date_str: str,
    output_dir: str,
    initial_active_rate_min: float = 0.5,
    initial_active_rate_max: float = 0.7,
    initial_availability_time_min_minutes: int = 0,
    initial_availability_time_max_minutes: int = 30,
    simulation_duration_hours: int = 24,
    random_seed: int = 42,
    verbose_logging: bool = False,
    enable_detailed_travel_time_analysis: bool = False,
    use_probabilistic_selection: bool = True,  # 軽症・中等症・死亡に確率的選択を適用
    enable_breaks: bool = False,  # 休憩機能の有効/無効を追加
    dispatch_strategy: str = 'closest',  # 追加
    strategy_config: Dict = None,  # 追加
    enable_visualization: bool = True,  # 可視化の有効/無効（軽量モード対応）
    enable_detailed_reports: bool = True  # 詳細レポートの有効/無効（軽量モード対応）
) -> None:
    """
    検証用シミュレーションを実行
    
    Args:
        use_probabilistic_selection: 軽症・中等症・死亡に確率的病院選択を適用するかのフラグ
                                   重症・重篤は常に決定論的選択（最寄りの3次救急病院）
    """
    
    # 乱数シードの設定
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    print("=" * 60)
    print("検証シミュレーション開始（最適化版）")
    print("=" * 60)
    
    # データキャッシュの初期化確認
    data_cache = get_emergency_data_cache()
    cache_info = data_cache.get_cache_info()
    if cache_info["cached"]:
        print(f"✓ データキャッシュ使用中: {cache_info['total_records']:,}件")
        print(f"  データ期間: {cache_info['date_range']['start']} ～ {cache_info['date_range']['end']}")
    else:
        print("初回データ読み込み中...")
        data_cache.load_data()
        print("データキャッシュ準備完了")
    
    # データの読み込み
    base_dir = CURRENT_PROJECT_DIR / "data" / "tokyo"
    
    # 救急隊と消防署の位置情報
    print("救急隊データを読み込み中...")
    firestation_csv_path = base_dir / "import" / "amb_place_master.csv"
    ambulance_data = pd.DataFrame()  # 初期化
    try:
        ambulance_data = pd.read_csv(firestation_csv_path, encoding='utf-8')
        print(f"情報: {firestation_csv_path} を utf-8 で読み込みました。")
    except UnicodeDecodeError:
        try:
            ambulance_data = pd.read_csv(firestation_csv_path, encoding='cp932')
            print(f"情報: {firestation_csv_path} を cp932 で読み込みました。")
        except Exception as e_cp932:
            print(f"エラー: {firestation_csv_path} の読み込みに失敗しました (utf-8, cp932)。エンコーディングを確認してください。 Error: {e_cp932}")
    except FileNotFoundError:
        print(f"エラー: {firestation_csv_path} が見つかりません。シミュレーションは消防署データなしで続行される可能性があります。")

    if not ambulance_data.empty:
        if 'special_flag' in ambulance_data.columns:
            original_count = len(ambulance_data)
            ambulance_data = ambulance_data[ambulance_data['special_flag'] == 1].copy()
            print(f"情報: 消防署データを special_flag == 1 でフィルタリングしました。(変更前: {original_count}件, 変更後: {len(ambulance_data)}件)")
            if ambulance_data.empty and original_count > 0:
                print(f"警告: {firestation_csv_path} から special_flag == 1 のデータが見つかりませんでした。")
        else:
            print(f"警告: {firestation_csv_path} に 'special_flag' カラムが見つかりません。消防署のフィルタリングは行われません。")
    elif ambulance_data.empty and os.path.exists(firestation_csv_path):
        print(f"警告: {firestation_csv_path} は存在しますが、読み込み後のデータが空です。ファイル内容やエンコーディングを確認してください。")
    
    # キャリブレーション済み移動時間行列の読み込み (フェーズ別)
    print("キャリブレーション済み移動時間行列を読み込み中...")
    travel_time_matrices = {}
    calibration_dir = base_dir / "calibration2"
    travel_time_stats_path = calibration_dir / 'travel_time_statistics_all_phases.json'

    if not os.path.exists(travel_time_stats_path):
        print(f"エラー: 移動時間統計ファイルが見つかりません: {travel_time_stats_path}。処理を中止します。")
        return

    with open(travel_time_stats_path, 'r', encoding='utf-8') as f:
        phase_stats_data = json.load(f)

    for phase in ['response', 'transport', 'return']:
        matrix_filename = None
        model_type_used = None
        
        # JSONファイルから各フェーズの最適モデルタイプを読み取り
        if phase in phase_stats_data and 'calibrated' in phase_stats_data[phase] and \
           'model_type' in phase_stats_data[phase]['calibrated']:
            
            model_type_used = phase_stats_data[phase]['calibrated']['model_type']
            if model_type_used == "uncalibrated":
                matrix_filename = f"uncalibrated_travel_time_{phase}.npy"
            elif model_type_used in ['linear', 'log']:
                matrix_filename = f"{model_type_used}_calibrated_{phase}.npy"
            else:
                print(f"警告: {phase} フェーズの校正モデルタイプが不明です: {model_type_used}")
                continue
        else:
            print(f"警告: {phase} フェーズの校正情報が {travel_time_stats_path} に見つかりません。")
            continue

        if matrix_filename:
            matrix_path = calibration_dir / matrix_filename
            if os.path.exists(matrix_path):
                print(f"  > '{phase}'フェーズの行列を読み込みます: {matrix_filename} (モデル: {model_type_used})")
                travel_time_matrices[phase] = np.load(matrix_path)
            else:
                print(f"  > エラー: 移動時間行列ファイルが見つかりません: {matrix_path}")

    if not travel_time_matrices or not all(p in travel_time_matrices for p in ['response', 'transport', 'return']):
        print(f"エラー: 必須の移動時間行列が読み込めませんでした。ロード済み: {list(travel_time_matrices.keys())}")
        return

    # 距離行列とグリッドマッピングの読み込み
    print("移動距離行列とグリッドマッピングを読み込み中...")
    with open(base_dir / "processed" / "travel_distance_matrix_res9.npy", 'rb') as f:
        travel_distance_matrix = np.load(f)
    with open(base_dir / "processed" / "grid_mapping_res9.json", 'r', encoding='utf-8') as f:
        grid_mapping = json.load(f)
        
    # 病院データの読み込み
    print("病院データを読み込み中...")
    hospital_data = pd.read_csv(base_dir / "import" / "hospital_master.csv", encoding='utf-8')
    hospital_data = hospital_data.rename(columns={'hospital_latitude': 'latitude', 'hospital_longitude': 'longitude'})
    
    # 病院データの検証
    print(f"読み込まれた病院データ: {len(hospital_data)}件")
    if 'genre_code' in hospital_data.columns:
        genre_counts = hospital_data['genre_code'].value_counts().sort_index()
        print("病院種別分布:")
        for genre_code, count in genre_counts.items():
            if genre_code == 1:
                genre_name = "3次救急"
            elif genre_code == 2:
                genre_name = "2次以下"
            else:
                genre_name = f"その他(code:{genre_code})"
            print(f"  {genre_name}: {count}件")
    
    # H3インデックスの生成
    hospital_h3_indices = []
    for idx, hospital in hospital_data.iterrows():
        if pd.notna(hospital['latitude']) and pd.notna(hospital['longitude']):
            h3_idx = h3.latlng_to_cell(hospital['latitude'], hospital['longitude'], 9)
            if h3_idx in grid_mapping:
                hospital_h3_indices.append(h3_idx)
    
    print(f"有効な病院H3インデックス: {len(hospital_h3_indices)}件")
    
    # 救急事案データ（最適化版）
    print("救急事案データ取得中...")
    
    # 日時範囲を計算
    try:
        sim_start_datetime = pd.to_datetime(target_date_str)  # YYYY-MM-DD HH:MM:SS を期待。時刻がなければ00:00:00
    except ValueError:
        print(f"エラー: target_date_str '{target_date_str}' は正しい日付形式 (YYYY-MM-DD) ではありません。処理を中止します。")
        return None

    end_datetime_sim = sim_start_datetime + pd.Timedelta(hours=simulation_duration_hours)
    
    # キャッシュから高速取得（前処理済み：日付変換、「その他」除外済み）
    calls_df = get_datetime_range_emergency_data(sim_start_datetime, end_datetime_sim)
    print(f"期間内の事案数: {len(calls_df)}件")
    
    # サービス時間パラメータの読み込み
    hierarchical_params_path = base_dir / "service_time_analysis" / "lognormal_parameters_hierarchical.json"
    standard_params_path = base_dir / "service_time_analysis" / "v1" / "lognormal_parameters.json"
    
    print(f"[DEBUG] USE_ENHANCED_GENERATOR = {USE_ENHANCED_GENERATOR}")
    print(f"[DEBUG] 階層的パラメータファイルパス: {hierarchical_params_path}")
    print(f"[DEBUG] 階層的パラメータファイル存在: {os.path.exists(hierarchical_params_path)}")
    print(f"[DEBUG] 標準パラメータファイルパス: {standard_params_path}")
    print(f"[DEBUG] 標準パラメータファイル存在: {os.path.exists(standard_params_path)}")
    
    if USE_ENHANCED_GENERATOR and os.path.exists(hierarchical_params_path):
        print("階層的パラメータを使用してServiceTimeGeneratorEnhancedを初期化")
        service_time_generator = ServiceTimeGeneratorEnhanced(hierarchical_params_path)
    elif os.path.exists(standard_params_path):
        print("標準パラメータを使用してServiceTimeGeneratorを初期化")
        service_time_generator = ServiceTimeGenerator(standard_params_path)
    else:
        raise FileNotFoundError("サービス時間パラメータファイルが見つかりません")
    
    # シミュレーション開始日時の設定
    sim_start_datetime = datetime.strptime(target_date_str, "%Y%m%d")
    
    # シミュレータの初期化
    simulator = ValidationSimulator(
        travel_time_matrices=travel_time_matrices,
        travel_distance_matrices={'default': travel_distance_matrix},
        grid_mapping=grid_mapping,
        service_time_generator=service_time_generator,
        hospital_h3_indices=hospital_h3_indices,
        hospital_data=hospital_data,
        use_probabilistic_selection=use_probabilistic_selection,  # 軽症・中等症の確率的選択制御
        enable_breaks=enable_breaks,
        dispatch_strategy=dispatch_strategy,  # 追加
        strategy_config=strategy_config  # 追加
    )
    
    # 出力ディレクトリの設定と分析機能の有効化
    # output_dirは呼び出し元から渡されるようになりました
    
    # 詳細分析機能を有効にする
    if enable_detailed_travel_time_analysis:
        simulator.enable_detailed_analysis(output_dir)
    
    # 救急隊の初期化
    simulator.initialize_ambulances(
        ambulance_data=ambulance_data,
        include_daytime_ambulances=False,
        initial_active_rate_range=(initial_active_rate_min, initial_active_rate_max),
        initial_availability_time_range_minutes=(initial_availability_time_min_minutes, initial_availability_time_max_minutes)
    )
    
    # 事案の追加
    simulator.add_emergency_calls(calls_df, sim_start_datetime)
    
    # シミュレーションの実行
    report = simulator.run(end_time=simulation_duration_hours * 3600, verbose=verbose_logging)
    
    # 直近隊選択統計をレポートに追加
    if report and hasattr(simulator, 'dispatch_strategy') and simulator.dispatch_strategy:
        try:
            dispatch_stats = simulator.dispatch_strategy.get_dispatch_statistics()
            if dispatch_stats:
                report['dispatch_statistics'] = dispatch_stats
        except Exception as e:
            # エラーが発生してもシミュレーション結果は保存
            pass
    
    # レポートの保存（軽量モード対応）
    if report:
        os.makedirs(output_dir, exist_ok=True)
        
        # 軽量モード時はrun番号付きで保存（baseline_comparison.pyで管理）
        if not enable_detailed_reports:
            # batchディレクトリ内のrun番号を取得
            existing_reports = [f for f in os.listdir(output_dir) if f.startswith('simulation_report_run') and f.endswith('.json')]
            run_number = len(existing_reports) + 1
            report_filename = f"simulation_report_run{run_number}.json"
        else:
            report_filename = "simulation_report.json"
        
        with open(f"{output_dir}/{report_filename}", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 可視化の作成（軽量モード時はスキップ）
        if enable_visualization:
            create_simulation_visualizations_enhanced(report, output_dir)
            print(f"シミュレーション結果と可視化を {output_dir} に保存しました。")
        else:
            print(f"シミュレーション結果を {output_dir} に保存しました（可視化スキップ）。")
        
        # 分析結果のサマリー表示
        if enable_detailed_travel_time_analysis and 'travel_time_analysis' in report:
            print(f"\n=== 移動時間分析サマリー ===")
            print(f"総移動記録数: {report['travel_time_analysis']['total_travel_records']}")
            print(f"フェーズ別同一グリッド移動数:")
            for phase, count in report['travel_time_analysis']['same_grid_counts_by_phase'].items():
                print(f"  {phase}: {count}回")
            print(f"\n距離カテゴリ別統計:")
            for category, stats in report['travel_time_analysis']['distance_category_stats'].items():
                if stats['count'] > 0:
                    print(f"  {category}: {stats['count']}回, 平均時間: {stats['mean_time']:.1f}分")
            print(f"\nコンテキスト別統計:")
            for context, stats in report['travel_time_analysis']['context_breakdown'].items():
                print(f"  {context}: {stats['count']}回, 平均時間: {stats['mean_time']:.1f}分")
        
        # 休憩統計のサマリー表示
        if enable_breaks and 'break_statistics' in report:
            print(f"\n=== 休憩統計サマリー ===")
            break_stats = report['break_statistics']
            print(f"レストタイム:")
            print(f"  総取得回数: {break_stats['rest_time']['total_count']}")
            print(f"  取得理由内訳:")
            for reason, count in break_stats['rest_time']['by_reason'].items():
                reason_labels = {
                    'lunch_period': '昼食(11-13時)',
                    'lunch_extended': '昼食延長(15時)',
                    'dinner_period': '夕食(17-19時)',
                    'dinner_extended': '夕食延長(20時)'
                }
                display_reason = reason_labels.get(reason, reason)
                print(f"    {display_reason}: {count}回")
            print(f"\n出場間インターバル:")
            print(f"  総取得回数: {break_stats['interval']['total_count']}")
            print(f"  取得前平均活動時間: {break_stats['interval']['average_activity_before']:.1f}時間")
            print(f"\n乗務員交代:")
            print(f"  総交代回数: {break_stats['crew_changes']['total_count']}")
    else:
        print("エラー: シミュレーションレポートが生成されませんでした。")


if __name__ == "__main__":
   # 例: 2023年1月1日のデータを対象に24時間シミュレーションを実行
   # 必要に応じて日付や期間を変更してください
   target_day = "2022-04-01"
   target_day_formatted = target_day.replace("-", "")  # "20230101"形式に変換
   duration = 720 # 168時間 = 7日間, 720時間 = 30日間, 8760時間 = 1年間
   
   # 詳細ログを出力する場合は verbose_logging=True に設定
   enable_verbose_logging = False # 病院選択ログを出力するためTrueに設定
   
   # 移動時間詳細分析機能を有効にする場合は True に設定
   enable_travel_time_analysis = True  # 移動時間の詳細分析を実行

   # 確率的選択を有効にするかどうかのフラグ（軽症・中等症・死亡のみ適用）
   use_probabilistic_hospital_selection = True  # True: 軽症・中等症・死亡に確率的選択, False: 全て決定論的選択

   # --- ディスパッチ戦略の設定 ---
   # dispatch_strategy = 'closest'  # 従来の最寄り戦略
   # dispatch_strategy = 'severity_based'  # 基本的な傷病度考慮戦略
   dispatch_strategy = 'advanced_severity'  # 高度な傷病度考慮戦略
   
   # 戦略固有の設定
   strategy_config = {}
   if dispatch_strategy == 'advanced_severity':
        # 設定プリセットから選択
        strategy_config = STRATEGY_CONFIGS['extreme']  # 'conservative', 'aggressive','extreme' から選択
        
        # または個別にカスタマイズ
        # strategy_config = {
        #     'mild_time_limit': 1080,        # 軽症の制限時間（秒）
        #     'mild_delay_threshold': 600,    # 軽症が優先する最小時間
        #     'moderate_time_limit': 900,     # 中等症の制限時間
        #     'high_utilization': 0.65,       # 繁忙期判定閾値
        #     'critical_utilization': 0.80,   # 緊急モード閾値
        # }

   # --- 救急隊の初期活動状態に関する設定 ---
   # シミュレーション開始時に活動中とする救急隊の割合の範囲 (例: 40%～60%)
   # 0.0に設定すると、全ての隊が待機状態から開始
   initial_active_min_percentage = 0.4  # 40%
   initial_active_max_percentage = 0.6  # 60%
   
   # 活動中の隊が利用可能になるまでの時間の範囲（分単位） (例: 0分～120分)
   initial_available_time_min = 0    # 0分
   initial_available_time_max = 120  # 120分   
   
   # --- バージョン管理付き出力ディレクトリの生成 ---
   output_base_dir = CURRENT_PROJECT_DIR / "data" / "tokyo" / "simulation_results"
   output_dir = get_versioned_output_dir(
       base_dir=str(output_base_dir), 
       date_str=target_day_formatted, 
       duration_hours=duration
   )
   
   print(f"成果物の出力先: {output_dir}")
   
   # phase_labels_jp グローバルスコープで定義 (run_validation_simulation 内のサマリー表示用)
   phase_labels_jp = {
       'dispatch_to_scene': '出場～現着', 'on_scene': '現場活動',
       'scene_to_hospital': '現発～病着', 'at_hospital': '病院滞在',
       'hospital_to_station': '帰署時間'
   }

   run_validation_simulation(
       target_date_str=target_day_formatted,
       output_dir=output_dir,
       initial_active_rate_min=initial_active_min_percentage,
       initial_active_rate_max=initial_active_max_percentage,
       initial_availability_time_min_minutes=initial_available_time_min,
       initial_availability_time_max_minutes=initial_available_time_max,
       simulation_duration_hours=duration,
       random_seed=42,
       verbose_logging=enable_verbose_logging,
       enable_detailed_travel_time_analysis=enable_travel_time_analysis,
       use_probabilistic_selection=use_probabilistic_hospital_selection,  # 軽症・中等症・死亡の確率的選択制御
       enable_breaks=False,  # 休憩機能の有効化
       dispatch_strategy=dispatch_strategy,
       strategy_config=strategy_config
   )