"""
baseline_comparison.py
複数ディスパッチ戦略の比較実験システム
"""

# OpenMPエラーの回避（ライブラリ競合対策）
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['WANDB_MODE'] = 'disabled'  # wandbを無効化
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from scipy import stats
import yaml
import sys
from pathlib import Path

import wandb

# matplotlibバックエンドを非インタラクティブに設定
import matplotlib
matplotlib.use('Agg')

# 日本語フォント設定
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['font.size'] = 12

# 現在のプロジェクトディレクトリ（05_Ambulance_RL）を取得
# ファイル構造: 05_Ambulance_RL/baseline_comparison.py
CURRENT_PROJECT_DIR = Path(__file__).resolve().parent
if str(CURRENT_PROJECT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_PROJECT_DIR))

# 後方互換性のため fix_dir も同じディレクトリを参照
fix_dir = CURRENT_PROJECT_DIR

# 親ディレクトリ（必要な場合のみ）
PROJECT_ROOT = CURRENT_PROJECT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ディスパッチ戦略のインポート
from dispatch_strategies import STRATEGY_CONFIGS, StrategyFactory, PPOStrategy

# 統一された傷病度定数をインポート
from constants import SEVERITY_GROUPS

# ============================================================
# 実験設定
# ============================================================
EXPERIMENT_CONFIG = {
    # 比較する戦略のリスト（ここで戦略を追加・削除）
    # モデル名を固定せず、PPO用のスロットを残し、パスだけ差し替えて使えるようにする
    # 既存の戦略も残しておく（比較材料）
    # 追加: ハバーシン距離ベースの直近隊（closest_haversine）
    'strategies': [
        # # 'closest',
        # 'closest_haversine',
        # 'closest_distance',
        # 'severity_based',
        # # 'advanced_severity',
        # 'second_ride',
        # 'mexclp',
        # 'ppo_agent',   # 従来PPO（ベースライン）
        'ppo_slot1',   # スロット1（パスを差し替えて利用）
        'ppo_slot2',   # スロット2
        'ppo_slot3',   # スロット3
        # 'ppo_slot4', # スロット4（必要ならコメント解除）
    ],
    
    # 各戦略の日本語表示名
    'strategy_labels': {
        'closest': '直近隊運用（時間）',
        'closest_haversine': '直近隊運用（ハバーシン距離）',
        'closest_distance': '直近隊運用（距離）',
        'severity_based': '傷病度考慮運用',
        'advanced_severity': '高度傷病度考慮運用',
        'second_ride': '2番目選択運用',  
        'ppo_agent': 'PPOエージェント（hybrid_intermediate_v4）',
        'mexclp': 'MEXCLP運用',
        # ↓ PPOスロット用の表示名
        'ppo_slot1': 'PPOエージェント（normal_unified_v14a）',
        'ppo_slot2': 'PPOエージェント（normal_unified_v14b）',
        'ppo_slot3': 'PPOエージェント（normal_unified_v14c）',
        'ppo_slot4': 'PPOエージェント（hybrid_unified_v6c）',
    },
    
    # 各戦略の色設定
    'strategy_colors': {
        'closest': '#3498db',           # 青
        'closest_haversine': '#2980b9', # 濃い青
        'closest_distance': '#1abc9c',  # ティール（青緑）
        'severity_based': '#e74c3c',    # 赤
        'advanced_severity': '#2ecc71', # 緑
        'second_ride': '#f39c12',       # オレンジ 
        'ppo_agent': '#9b59b6',         # 紫（従来PPO）
        'mexclp': '#e67e22',            # カロット
        # ↓ PPOスロット用の色
        'ppo_slot1': '#9b59b6',  # 紫
        'ppo_slot2': '#e91e63',  # ピンク
        'ppo_slot3': '#00bcd4',  # シアン
        'ppo_slot4': '#8e44ad',  # 予備色
    },
    
    # 各戦略の設定
    'strategy_configs': {
        'closest': {},
        'closest_haversine': {},  # ハバーシン距離ベースの最寄り戦略
        'closest_distance': {},  # 移動距離ベースの最寄り戦略（初期化時に行列を読み込む）
        'severity_based': {
            'coverage_radius_km': 5.0,
            'severe_conditions': SEVERITY_GROUPS['severe_conditions'],
            'mild_conditions': SEVERITY_GROUPS['mild_conditions'],
            'time_score_weight': 0.2,
            'coverage_loss_weight': 0.8,
            'mild_time_limit_sec': 1080,
            'moderate_time_limit_sec': 900
        },
        'advanced_severity': STRATEGY_CONFIGS['aggressive'],
        'second_ride': {
            'alternative_rank': 2,
            'enable_time_limit': False,
            'time_limit_seconds': 780
        },
        # 従来の単一PPO設定は残しておく（必要なら別実験で使用可能）
        'ppo_agent': {
            'model_path': str(fix_dir 
                              / 'reinforcement_learning' 
                              / 'experiments' 
                              / 'ppo_training' 
                              / 'hybrid_intermediate_v4' 
                              / 'checkpoints' 
                              / 'best_model.pth'),
            'config_path': str(fix_dir 
                               / 'reinforcement_learning' 
                               / 'experiments' 
                               / 'ppo_training' 
                               / 'hybrid_intermediate_v4' 
                               / 'configs' 
                               / 'config.yaml'),
            'hybrid_mode': False,
            'severe_conditions': ['重症', '重篤', '死亡'],
            'mild_conditions': ['軽症', '中等症'],
        },
        # ★★★ 以下、PPOスロット用の設定（パスを差し替えて利用）★★★
        'ppo_slot1': {
            'model_path': str(
                fix_dir
                / 'reinforcement_learning'
                / 'experiments'
                / 'ppo_training'
                / 'normal_unified_v14a'  # ←ここを差し替えてください
                / 'checkpoints'
                / 'best_model.pth'
            ),
            'config_path': str(
                fix_dir
                / 'reinforcement_learning'
                / 'experiments'
                / 'ppo_training'
                / 'normal_unified_v14a'  # ←ここを差し替えてください
                / 'configs'
                / 'config.yaml'
            ),
            'hybrid_mode': False,
            'severe_conditions': ['重症', '重篤', '死亡'],
            'mild_conditions': ['軽症', '中等症'],
        },
        'ppo_slot2': {
            'model_path': str(
                fix_dir
                / 'reinforcement_learning'
                / 'experiments'
                / 'ppo_training'
                / 'normal_unified_v14b'  # ←ここを差し替えてください
                / 'checkpoints'
                / 'best_model.pth'
            ),
            'config_path': str(
                fix_dir
                / 'reinforcement_learning'
                / 'experiments'
                / 'ppo_training'
                / 'normal_unified_v14b'  # ←ここを差し替えてください
                / 'configs'
                / 'config.yaml'
            ),
            'hybrid_mode': False,
            'severe_conditions': ['重症', '重篤', '死亡'],
            'mild_conditions': ['軽症', '中等症'],
        },
        'ppo_slot3': {
            'model_path': str(
                fix_dir
                / 'reinforcement_learning'
                / 'experiments'
                / 'ppo_training'
                / 'normal_unified_v14c'  # ←ここを差し替えてください
                / 'checkpoints'
                / 'best_model.pth'
            ),
            'config_path': str(
                fix_dir
                / 'reinforcement_learning'
                / 'experiments'
                / 'ppo_training'
                / 'normal_unified_v14c'  # ←ここを差し替えてください
                / 'configs'
                / 'config.yaml'
            ),
            'hybrid_mode': False,
            'severe_conditions': ['重症', '重篤', '死亡'],
            'mild_conditions': ['軽症', '中等症'],
        },
        # 4本目用の予備設定（パスは仮置き、必要時にコメント解除）
        'ppo_slot4': {
            'model_path': str(
                fix_dir
                / 'reinforcement_learning'
                / 'experiments'
                / 'ppo_training'
                / 'hybrid_unified_v6c_quick'  # ←ここを差し替えてください
                / 'checkpoints'
                / 'best_model.pth'
            ),
            'config_path': str(
                fix_dir
                / 'reinforcement_learning'
                / 'experiments'
                / 'ppo_training'
                / 'hybrid_unified_v6c_quick'  # ←ここを差し替えてください
                / 'configs'
                / 'config.yaml'
            ),
            'hybrid_mode': True,
            'severe_conditions': ['重症', '重篤', '死亡'],
            'mild_conditions': ['軽症', '中等症'],
        },
        'mexclp': {
            'busy_fraction': 0.3,
            'time_threshold_seconds': 780 # 13分
        }
        }
    }

# ============================================================
# テスト期間データと季節・象限判定
# ============================================================

# テスト期間の週次データ（件数・重症率）
# Excelデータから抽出した6期間のデータ
TEST_PERIOD_DATA = {
    '20240204': {'weekly_count': 10739, 'severe_rate': 0.0824},  # 2024/2/4週
    '20240331': {'weekly_count': 10232, 'severe_rate': 0.0738},  # 2024/3/31週
    '20240505': {'weekly_count': 10000, 'severe_rate': 0.0620},  # 2024/5/5週（推定）
    '20240630': {'weekly_count': 12000, 'severe_rate': 0.0550},  # 2024/6/30週（推定）
    '20240721': {'weekly_count': 13000, 'severe_rate': 0.0560},  # 2024/7/21週（推定）
    '20241222': {'weekly_count': 13565, 'severe_rate': 0.0738},  # 2024/12/22週
}


def get_season(date_str: str) -> str:
    """
    日付から季節を判定
    
    Args:
        date_str: YYYYMMDD形式の日付文字列
    
    Returns:
        季節（春/夏/秋/冬）
    """
    month = int(date_str[4:6])
    if month in [3, 4, 5]:
        return "春"
    elif month in [6, 7, 8]:
        return "夏"
    elif month in [9, 10, 11]:
        return "秋"
    else:
        return "冬"


def get_quadrant(date_str: str) -> str:
    """
    日付から象限を判定
    
    閾値：
    - 件数: 11000件/週
    - 重症率: 6.7%
    
    Args:
        date_str: YYYYMMDD形式の日付文字列
    
    Returns:
        象限（高件数×高重症/高件数×低重症/低件数×高重症/低件数×低重症）
    """
    # テスト期間データから取得（完全一致を探す）
    period_data = TEST_PERIOD_DATA.get(date_str)
    
    if period_data is None:
        # 完全一致がない場合、最も近い日付を探す
        target_date = datetime.strptime(date_str, '%Y%m%d')
        closest_key = None
        min_diff = float('inf')
        
        for key in TEST_PERIOD_DATA.keys():
            key_date = datetime.strptime(key, '%Y%m%d')
            diff = abs((target_date - key_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_key = key
        
        if closest_key and min_diff <= 7:  # 1週間以内なら採用
            period_data = TEST_PERIOD_DATA[closest_key]
        else:
            return "不明"
    
    weekly_count = period_data['weekly_count']
    severe_rate = period_data['severe_rate']
    
    # 閾値判定
    high_count = weekly_count >= 11000
    high_severe = severe_rate >= 0.067
    
    if high_count and high_severe:
        return "高件数×高重症"
    elif high_count and not high_severe:
        return "高件数×低重症"
    elif not high_count and high_severe:
        return "低件数×高重症"
    else:
        return "低件数×低重症"


# ============================================================
# PPOパラメータ抽出
# ============================================================

# 追跡する主要パラメータのキー
PPO_KEY_PARAMS = [
    ('state_encoding.top_k', 'top_k'),
    ('hybrid_mode.enabled', 'hybrid_mode'),
    ('reward.unified.time_weight', 'time_weight'),
    ('reward.unified.coverage_weight', 'coverage_weight'),
    ('reward.unified.coverage_penalty_scale', 'coverage_penalty_scale'),
    ('ppo.entropy_coef', 'entropy_coef'),
    ('ppo.n_episodes', 'n_episodes'),
]


def get_nested_value(data: dict, key_path: str, default=None):
    """
    ネストした辞書から値を取得
    
    Args:
        data: 辞書
        key_path: ドット区切りのキーパス（例: 'reward.unified.time_weight'）
        default: デフォルト値
    
    Returns:
        取得した値、またはデフォルト値
    """
    keys = key_path.split('.')
    value = data
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def extract_ppo_params(config_path: str) -> Dict[str, any]:
    """
    PPOのconfig.yamlから主要パラメータを抽出
    
    Args:
        config_path: config.yamlのパス
    
    Returns:
        パラメータ辞書（キー: パラメータ名、値: 設定値）
        読み込みに失敗した場合は空の辞書
    """
    params = {}
    
    if not config_path:
        return params
    
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"  警告: config.yamlが見つかりません: {config_path}")
        return params
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        for key_path, param_name in PPO_KEY_PARAMS:
            value = get_nested_value(config, key_path)
            params[param_name] = value
        
    except Exception as e:
        print(f"  警告: config.yaml読み込みエラー: {e}")
    
    return params


# ============================================================
# CSVマスター追記機能
# ============================================================

# CSVカラム定義
CSV_COLUMNS = [
    # メタ情報
    '実験日時',
    '実験名',
    'テスト開始日',
    'テスト終了日',
    '季節',
    '象限',
    # 戦略情報
    '戦略ID',
    '戦略表示名',
    'モデルパス',
    # 結果指標
    '重症RT',
    '重症RT_std',
    '全体RT',
    '全体RT_std',
    '軽症RT',
    '軽症RT_std',
    '6分率_重症',
    '6分率_重症_std',
    '6分率_全体',
    '6分率_全体_std',
    '13分率_全体',
    '13分率_全体_std',
    '直近隊率_全体',
    '直近隊率_重症',
    '直近隊率_軽症',
    # PPOパラメータ
    'top_k',
    'hybrid_mode',
    'time_weight',
    'coverage_weight',
    'coverage_penalty_scale',
    'entropy_coef',
    'n_episodes',
    # 備考
    '備考',
]


def append_to_master_csv(
    aggregated: Dict,
    experiment_metadata: Dict,
    output_dir: str,
    master_csv_path: Optional[str] = None
):
    """
    実験結果をマスターCSVに追記
    
    Args:
        aggregated: 集約された統計情報
        experiment_metadata: 実験メタデータ
        output_dir: 出力ディレクトリ（デフォルトのCSV保存先）
        master_csv_path: マスターCSVのパス（指定しない場合はoutput_dirの親に保存）
    """
    # マスターCSVのパス決定
    if master_csv_path is None:
        # baseline_comparison.pyと同じディレクトリ（fix_dir直下）に保存
        master_csv_path = fix_dir / 'all_experiment_results.csv'
    else:
        master_csv_path = Path(master_csv_path)
    
    # 実験メタデータから情報取得
    config = experiment_metadata['configuration']
    experiment_name = experiment_metadata['experiment_name']
    timestamp = experiment_metadata['timestamp']
    start_date = config['start_date']
    end_date = config['end_date']
    
    # 季節・象限の判定
    season = get_season(start_date)
    quadrant = get_quadrant(start_date)
    
    # 戦略設定
    strategies = config['strategies']
    strategy_configs = config.get('strategy_configs', {})
    strategy_labels = EXPERIMENT_CONFIG['strategy_labels']
    
    # 各戦略の結果をリストに追加
    rows = []
    
    for strategy in strategies:
        data = aggregated.get(strategy, {})
        strategy_config = strategy_configs.get(strategy, {})
        
        # モデルパス（PPOの場合のみ）
        model_path = strategy_config.get('model_path', '')
        config_path = strategy_config.get('config_path', '')
        
        # PPOパラメータの抽出（PPO戦略の場合のみ）
        ppo_params = {}
        if strategy.startswith('ppo_') and config_path:
            ppo_params = extract_ppo_params(config_path)
        
        # 行データの構築
        row = {
            # メタ情報
            '実験日時': timestamp,
            '実験名': experiment_name,
            'テスト開始日': start_date,
            'テスト終了日': end_date,
            '季節': season,
            '象限': quadrant,
            # 戦略情報
            '戦略ID': strategy,
            '戦略表示名': strategy_labels.get(strategy, strategy),
            'モデルパス': model_path,
            # 結果指標
            '重症RT': data.get('response_time_severe', {}).get('mean', ''),
            '重症RT_std': data.get('response_time_severe', {}).get('std', ''),
            '全体RT': data.get('response_time_overall', {}).get('mean', ''),
            '全体RT_std': data.get('response_time_overall', {}).get('std', ''),
            '軽症RT': data.get('response_time_mild', {}).get('mean', ''),
            '軽症RT_std': data.get('response_time_mild', {}).get('std', ''),
            '6分率_重症': data.get('threshold_6min_severe', {}).get('mean', ''),
            '6分率_重症_std': data.get('threshold_6min_severe', {}).get('std', ''),
            '6分率_全体': data.get('threshold_6min', {}).get('mean', ''),
            '6分率_全体_std': data.get('threshold_6min', {}).get('std', ''),
            '13分率_全体': data.get('threshold_13min', {}).get('mean', ''),
            '13分率_全体_std': data.get('threshold_13min', {}).get('std', ''),
            '直近隊率_全体': data.get('closest_dispatch_rate', {}).get('mean', ''),
            '直近隊率_重症': data.get('closest_dispatch_rate_by_severity', {}).get('severe', {}).get('mean', ''),
            '直近隊率_軽症': data.get('closest_dispatch_rate_by_severity', {}).get('mild', {}).get('mean', ''),
            # PPOパラメータ（PPO以外は空欄）
            'top_k': ppo_params.get('top_k', ''),
            'hybrid_mode': ppo_params.get('hybrid_mode', ''),
            'time_weight': ppo_params.get('time_weight', ''),
            'coverage_weight': ppo_params.get('coverage_weight', ''),
            'coverage_penalty_scale': ppo_params.get('coverage_penalty_scale', ''),
            'entropy_coef': ppo_params.get('entropy_coef', ''),
            'n_episodes': ppo_params.get('n_episodes', ''),
            # 備考
            '備考': '',
        }
        rows.append(row)
    
    # DataFrameに変換
    df_new = pd.DataFrame(rows, columns=CSV_COLUMNS)
    
    # 既存CSVがあれば追記、なければ新規作成
    if master_csv_path.exists():
        df_existing = pd.read_csv(master_csv_path, encoding='utf-8-sig')
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    # CSV保存（BOM付きUTF-8でExcel互換性確保）
    df_combined.to_csv(master_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n実験結果をマスターCSVに追記しました: {master_csv_path}")
    print(f"  追加行数: {len(rows)}")
    print(f"  累計行数: {len(df_combined)}")


def register_ppo_slots():
    """
    EXPERIMENT_CONFIGで指定されたPPOスロット名をStrategyFactoryに登録する。
    名称を固定しておけば、model_pathだけ差し替えて複数モデルを比較可能。
    """
    for name in EXPERIMENT_CONFIG['strategies']:
        if name.startswith('ppo_') and name not in StrategyFactory.list_available_strategies():
            StrategyFactory.register_strategy(name, PPOStrategy)


def flatten_dict(d, parent_key='', sep='.'):
    """ネストした辞書をフラットな辞書に変換する（wandb用）"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def generate_random_date_in_range(start_date: str, end_date: str) -> str:
    """
    指定期間内からランダムな日付を選択
    
    Args:
        start_date: 開始日 (YYYYMMDD形式)
        end_date: 終了日 (YYYYMMDD形式)
    
    Returns:
        ランダムに選択された日付 (YYYYMMDD形式)
    """
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    
    # 日数差を計算
    delta_days = (end_dt - start_dt).days
    
    if delta_days < 0:
        raise ValueError(f"開始日({start_date})が終了日({end_date})より後です")
    
    # ランダムに日数を選択
    random_days = np.random.randint(0, delta_days + 1)
    
    # 選択された日付を計算
    selected_dt = start_dt + timedelta(days=random_days)
    
    return selected_dt.strftime('%Y%m%d')

def run_comparison_experiment(
    start_date: str,
    end_date: str,
    episode_duration_hours: int = 24,
    num_runs: int = 100,
    output_base_dir: str = None,
    wandb_project: str = "ambulance-dispatch-simulation",
    experiment_name: str = None
):
    """
    複数日にわたるランダムサンプリング実験
    
    Args:
        start_date: 期間の開始日（YYYYMMDD形式）
        end_date: 期間の終了日（YYYYMMDD形式）
        episode_duration_hours: 1エピソードの長さ（時間）
        num_runs: 各戦略の実行回数（ランダムサンプリング）
        output_base_dir: 結果出力ベースディレクトリ
        wandb_project: wandbのプロジェクト名
        experiment_name: カスタム実験名（オプション）
    """
    # PPOスロット名を事前登録（Unknown strategy エラー防止）
    register_ppo_slots()

    # output_base_dirがNoneの場合は絶対パスを設定
    # 従来: プロジェクト直下の data/tokyo/experiments
    # 今回: 修正版ディレクトリ配下の data/tokyo/experiments に分離
    if output_base_dir is None:
        output_base_dir = str(fix_dir / "data" / "tokyo" / "experiments")
    
    # 実験ディレクトリの作成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if experiment_name:
        experiment_dir_name = f"{experiment_name}_{timestamp}"
    else:
        strategies_count = len(EXPERIMENT_CONFIG['strategies'])
        experiment_dir_name = f"comparison_{start_date}-{end_date}_{episode_duration_hours}h_{num_runs}runs_{timestamp}"
    
    experiment_dir = os.path.join(output_base_dir, experiment_dir_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 日数を計算
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    total_days = (end_dt - start_dt).days + 1
    
    print("=" * 80)
    print("ディスパッチ戦略比較実験 (ランダムサンプリング)")
    print(f"実験ディレクトリ: {experiment_dir}")
    print(f"期間: {start_date} - {end_date} ({total_days}日間)")
    print(f"エピソード長: {episode_duration_hours}時間")
    print(f"実行回数: 各戦略 {num_runs}回")
    print(f"比較戦略: {', '.join([EXPERIMENT_CONFIG['strategy_labels'][s] for s in EXPERIMENT_CONFIG['strategies']])}")
    print("=" * 80)
    
    # 実験メタデータの保存
    experiment_metadata = {
        'experiment_name': experiment_dir_name,
        'timestamp': timestamp,
        'configuration': {
            'start_date': start_date,
            'end_date': end_date,
            'total_days': total_days,
            'episode_duration_hours': episode_duration_hours,
            'num_runs': num_runs,
            'strategies': EXPERIMENT_CONFIG['strategies'],
            'strategy_configs': {
                strategy: EXPERIMENT_CONFIG['strategy_configs'][strategy]
                for strategy in EXPERIMENT_CONFIG['strategies']
            }
        },
        'environment': {
            'python_version': sys.version,
            'random_seed_base': 42
        }
    }
    
    metadata_path = os.path.join(experiment_dir, 'experiment_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_metadata, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"実験メタデータを保存: {metadata_path}\n")
    
    # wandbの初期設定
    try:
        wandb.login()
        print(f"wandbログイン成功: プロジェクト '{wandb_project}' に接続します")
    except Exception as e:
        print(f"警告: wandbログインに失敗しました。ローカルモードで実行します: {e}")
        os.environ['WANDB_MODE'] = 'disabled'
    
    # validation_simulation.pyのrun_validation_simulation関数をインポート
    from validation_simulation import run_validation_simulation
    
    # 設定から戦略リストを取得
    strategies = EXPERIMENT_CONFIG['strategies']
    strategy_configs = EXPERIMENT_CONFIG['strategy_configs']
    
    # 実験結果格納用（動的に初期化）
    results = {strategy: [] for strategy in strategies}
    
    for strategy in strategies:
        print(f"\n戦略: {EXPERIMENT_CONFIG['strategy_labels'][strategy]} ({strategy})")
        print("-" * 40)
        
        # 戦略ごとのbatchディレクトリを実験ディレクトリ内に作成
        strategy_batch_dir = os.path.join(experiment_dir, f"{strategy}_batch")
        os.makedirs(strategy_batch_dir, exist_ok=True)
        
        # 軽量モード判定
        # 比較実験では各runごとの詳細可視化は不要で、集約結果のみを使うため、
        # 常に軽量モード（run番号付きJSON + 可視化スキップ）とする。
        is_lightweight_mode = True
        if is_lightweight_mode:
            print(f"  軽量モード: 個別グラフ生成をスキップします")
        
        for run_idx in range(num_runs):
            # ランダム日付選択
            selected_date = generate_random_date_in_range(start_date, end_date)
            
            print(f"  実行 {run_idx + 1}/{num_runs}... (日付: {selected_date})")
            
            run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            run_name = f"{strategy}-{selected_date}-run{run_idx + 1}"
            
            plt.close('all')
            
            current_strategy_config = strategy_configs.get(strategy, {})
            config_for_wandb = {
                "start_date": start_date,
                "end_date": end_date,
                "selected_date": selected_date,
                "episode_duration_hours": episode_duration_hours,
                "num_runs": num_runs,
                "run_index": run_idx + 1,
                "random_seed": 42 + run_idx,
                "dispatch_strategy": strategy,
                **flatten_dict(current_strategy_config, parent_key='strategy_params')
            }
            
            # 軽量モード：batchディレクトリに直接保存
            output_dir = strategy_batch_dir

            run = None
            if os.environ.get('WANDB_MODE') == 'disabled':
                pass  # ローカルモード
            else:
                try:
                    run = wandb.init(
                        project=wandb_project,
                        config=config_for_wandb,
                        group=f"{strategy}-{start_date}-{end_date}",
                        name=run_name,
                        job_type="simulation",
                        tags=["baseline", strategy, "random_sampling"],
                        settings=wandb.Settings(init_timeout=120),
                        resume="allow"
                    )
                except Exception as e:
                    print(f"  - wandb連携エラー(初期化): {e}")
                    run = None
            
            # シミュレーション実行（軽量モード対応）
            run_validation_simulation(
                target_date_str=selected_date,
                output_dir=output_dir,
                simulation_duration_hours=episode_duration_hours,
                random_seed=42 + run_idx,
                verbose_logging=False,
                enable_visualization=not is_lightweight_mode,
                enable_detailed_reports=not is_lightweight_mode,
                dispatch_strategy=strategy,
                strategy_config=current_strategy_config
            )

            # レポート読み込み（run番号付きファイル名）
            report_filename = f"simulation_report_run{run_idx + 1}.json"
            report_path = os.path.join(output_dir, report_filename)
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    # 実行情報を追加
                    report['run_metadata'] = {
                        'run_index': run_idx + 1,
                        'selected_date': selected_date,
                        'random_seed': 42 + run_idx
                    }
                    results[strategy].append(report)

                    if run is not None:
                        unified_metrics = {}
                        rt_stats = report.get('response_times', {})
                        unified_metrics['charts/response_time_mean'] = rt_stats.get('overall', {}).get('mean', 0)
                        rt_by_severity = rt_stats.get('by_severity', {})
                        unified_metrics['charts/response_time_mild_mean'] = rt_by_severity.get('軽症', {}).get('mean', 0)
                        unified_metrics['charts/response_time_moderate_mean'] = rt_by_severity.get('中等症', {}).get('mean', 0)
                        unified_metrics['charts/response_time_severe_mean'] = rt_by_severity.get('重症', {}).get('mean', 0)
                        unified_metrics['charts/response_time_critical_mean'] = rt_by_severity.get('重篤', {}).get('mean', 0)
                        th_by_severity = report.get('threshold_performance', {}).get('by_severity', {}).get('6_minutes', {})
                        # rateは0-100スケール（パーセント）で保存されているため、0-1スケールに変換
                        unified_metrics['charts/response_time_severe_under_6min_rate'] = th_by_severity.get('重症', {}).get('rate', 0) / 100
                        wandb.log(unified_metrics)
                        wandb.log({"full_report": report})
                        print(f"  - wandbに統一されたメトリクスを記録しました。 (Run Name: {run_name})")
                    else:
                        print(f"  - ローカルモードで実行完了 (Run Name: {run_name})")

            except FileNotFoundError:
                print(f"  - エラー: レポートファイルが見つかりません: {report_path}")
                if run is not None:
                    wandb.log({"error": "report_not_found"})
            finally:
                if run is not None:
                    wandb.finish()
    
    # 結果の集約と分析（実験ディレクトリに保存）
    print("\n" + "=" * 80)
    print("実験結果の集約と分析")
    print("=" * 80)
    
    # 効率的な集約
    aggregated = aggregate_results_efficiently(results, experiment_dir, num_runs)
    
    # 収束分析の可視化
    visualize_convergence(results, experiment_dir)
    
    # 従来の可視化（集約データを使用）
    visualize_comparison(aggregated, experiment_dir)
    
    # 詳細サマリーレポート（実験設定込み）
    create_detailed_summary_report(aggregated, experiment_dir, experiment_metadata)
    
    # ★★★ マスターCSVへの追記 ★★★
    append_to_master_csv(aggregated, experiment_metadata, experiment_dir)
    
    print(f"\n実験完了！")
    print(f"結果は以下のディレクトリに保存されました: {experiment_dir}")
    
    return aggregated, experiment_dir

def analyze_results(results: Dict[str, List]) -> Dict:
    """実験結果を分析"""
    analysis = {}
    
    for strategy in results.keys():
        strategy_results = results[strategy]
        
        # 応答時間の統計
        response_times_all = []
        response_times_severe = []
        response_times_mild = []
        
        # 閾値達成率
        threshold_6min_rates = []
        threshold_13min_rates = []
        threshold_6min_severe_rates = []
        
        for report in strategy_results:
            # 全体の応答時間
            if 'response_times' in report and 'overall' in report['response_times']:
                rt_mean = report['response_times']['overall']['mean']
                response_times_all.append(rt_mean)
            
            # 傷病度別応答時間
            if 'by_severity' in report['response_times']:
                # 重症系
                for sev in ['重症', '重篤', '死亡']:
                    if sev in report['response_times']['by_severity']:
                        response_times_severe.append(
                            report['response_times']['by_severity'][sev]['mean']
                        )
                # 軽症系
                for sev in ['軽症', '中等症']:
                    if sev in report['response_times']['by_severity']:
                        response_times_mild.append(
                            report['response_times']['by_severity'][sev]['mean']
                        )
            
            # 閾値達成率
            if 'threshold_performance' in report:
                threshold_6min_rates.append(
                    report['threshold_performance']['6_minutes']['rate']
                )
                threshold_13min_rates.append(
                    report['threshold_performance']['13_minutes']['rate']
                )
                
                # 重症系の6分達成率
                if 'by_severity' in report['threshold_performance']:
                    severe_6min_rates = []
                    for sev in ['重症', '重篤']:
                        if sev in report['threshold_performance']['by_severity']['6_minutes']:
                            severe_6min_rates.append(
                                report['threshold_performance']['by_severity']['6_minutes'][sev]['rate']
                            )
                    if severe_6min_rates:
                        threshold_6min_severe_rates.append(np.mean(severe_6min_rates))
        
        # 統計値の計算
        analysis[strategy] = {
            'response_time_overall': {
                'mean': np.mean(response_times_all),
                'std': np.std(response_times_all),
                'values': response_times_all
            },
            'response_time_severe': {
                'mean': np.mean(response_times_severe) if response_times_severe else 0,
                'std': np.std(response_times_severe) if response_times_severe else 0,
                'values': response_times_severe
            },
            'response_time_mild': {
                'mean': np.mean(response_times_mild) if response_times_mild else 0,
                'std': np.std(response_times_mild) if response_times_mild else 0,
                'values': response_times_mild
            },
            'threshold_6min': {
                'mean': np.mean(threshold_6min_rates),
                'std': np.std(threshold_6min_rates),
                'values': threshold_6min_rates
            },
            'threshold_13min': {
                'mean': np.mean(threshold_13min_rates),
                'std': np.std(threshold_13min_rates),
                'values': threshold_13min_rates
            },
            'threshold_6min_severe': {
                'mean': np.mean(threshold_6min_severe_rates) if threshold_6min_severe_rates else 0,
                'std': np.std(threshold_6min_severe_rates) if threshold_6min_severe_rates else 0,
                'values': threshold_6min_severe_rates
            }
        }
    
    return analysis

def aggregate_results_efficiently(results: Dict[str, List], 
                                  output_dir: str,
                                  num_runs: int) -> Dict:
    """
    大量実行結果を効率的に集約
    
    Args:
        results: 戦略ごとの結果リスト
        output_dir: 出力ディレクトリ
        num_runs: 実行回数
    
    Returns:
        集約された統計情報
    """
    print(f"\n結果集約中: {num_runs}回の実行結果を統計処理")
    
    aggregated = {}
    
    for strategy, reports in results.items():
        print(f"\n戦略: {strategy} - {len(reports)}件の結果を集約")
        
        # 応答時間データの収集
        all_response_times = []
        severe_response_times = []
        mild_response_times = []
        
        threshold_6min_rates = []
        threshold_13min_rates = []
        threshold_6min_severe_rates = []
        
        # ★★★ 直近隊選択統計の収集（合計ベース） ★★★
        total_closest_dispatches = 0
        total_dispatches = 0
        total_closest_severe = 0
        total_dispatches_severe = 0
        total_closest_mild = 0
        total_dispatches_mild = 0
        total_closest_other = 0
        total_dispatches_other = 0
        
        for idx, report in enumerate(reports):
            # 基本統計
            if 'response_times' in report and 'overall' in report['response_times']:
                all_response_times.append(report['response_times']['overall']['mean'])
            
            # 傷病度別
            if 'by_severity' in report.get('response_times', {}):
                for sev in ['重症', '重篤', '死亡']:
                    if sev in report['response_times']['by_severity']:
                        severe_response_times.append(
                            report['response_times']['by_severity'][sev]['mean']
                        )
                for sev in ['軽症', '中等症']:
                    if sev in report['response_times']['by_severity']:
                        mild_response_times.append(
                            report['response_times']['by_severity'][sev]['mean']
                        )
            
            # 閾値達成率
            if 'threshold_performance' in report:
                threshold_6min_rates.append(
                    report['threshold_performance']['6_minutes']['rate']
                )
                threshold_13min_rates.append(
                    report['threshold_performance']['13_minutes']['rate']
                )
                
                # 重症系の6分達成率
                if 'by_severity' in report['threshold_performance']:
                    severe_6min_rates = []
                    for sev in ['重症', '重篤']:
                        if sev in report['threshold_performance']['by_severity']['6_minutes']:
                            severe_6min_rates.append(
                                report['threshold_performance']['by_severity']['6_minutes'][sev]['rate']
                            )
                    if severe_6min_rates:
                        threshold_6min_severe_rates.append(np.mean(severe_6min_rates))
            
            # ★★★ 直近隊選択統計の収集（合計ベース） ★★★
            if 'dispatch_statistics' in report:
                try:
                    dispatch_stats = report['dispatch_statistics']
                    if isinstance(dispatch_stats, dict):
                        # 全体の統計を合計
                        run_total = dispatch_stats.get('total_dispatches', 0)
                        run_closest = dispatch_stats.get('closest_dispatches', 0)
                        if run_total > 0:
                            total_dispatches += run_total
                            total_closest_dispatches += run_closest
                        
                        # 傷病度別統計を合計
                        by_severity = dispatch_stats.get('by_severity', {})
                        if isinstance(by_severity, dict):
                            # 重症系
                            severe_data = by_severity.get('severe', {})
                            if isinstance(severe_data, dict):
                                severe_total = severe_data.get('total', 0)
                                severe_closest = severe_data.get('closest', 0)
                                if severe_total > 0:
                                    total_dispatches_severe += severe_total
                                    total_closest_severe += severe_closest
                            
                            # 軽症系
                            mild_data = by_severity.get('mild', {})
                            if isinstance(mild_data, dict):
                                mild_total = mild_data.get('total', 0)
                                mild_closest = mild_data.get('closest', 0)
                                if mild_total > 0:
                                    total_dispatches_mild += mild_total
                                    total_closest_mild += mild_closest
                            
                            # その他
                            other_data = by_severity.get('other', {})
                            if isinstance(other_data, dict):
                                other_total = other_data.get('total', 0)
                                other_closest = other_data.get('closest', 0)
                                if other_total > 0:
                                    total_dispatches_other += other_total
                                    total_closest_other += other_closest
                except (KeyError, TypeError, AttributeError) as e:
                    # 統計データの読み込みエラーを無視（ログ出力はしない）
                    pass
        
        # 統計計算（信頼区間も含む）
        aggregated[strategy] = {
            'response_time_overall': {
                'mean': np.mean(all_response_times),
                'std': np.std(all_response_times),
                'median': np.median(all_response_times),
                'ci_95': stats.t.interval(0.95, len(all_response_times)-1,
                                         loc=np.mean(all_response_times),
                                         scale=stats.sem(all_response_times)) if len(all_response_times) > 1 else (0, 0),
                'min': np.min(all_response_times),
                'max': np.max(all_response_times),
                'values': all_response_times
            },
            'response_time_severe': {
                'mean': np.mean(severe_response_times) if severe_response_times else 0,
                'std': np.std(severe_response_times) if severe_response_times else 0,
                'median': np.median(severe_response_times) if severe_response_times else 0,
                'values': severe_response_times
            },
            'response_time_mild': {
                'mean': np.mean(mild_response_times) if mild_response_times else 0,
                'std': np.std(mild_response_times) if mild_response_times else 0,
                'median': np.median(mild_response_times) if mild_response_times else 0,
                'values': mild_response_times
            },
            'threshold_6min': {
                'mean': np.mean(threshold_6min_rates),
                'std': np.std(threshold_6min_rates),
                'values': threshold_6min_rates
            },
            'threshold_13min': {
                'mean': np.mean(threshold_13min_rates),
                'std': np.std(threshold_13min_rates),
                'values': threshold_13min_rates
            },
            'threshold_6min_severe': {
                'mean': np.mean(threshold_6min_severe_rates) if threshold_6min_severe_rates else 0,
                'std': np.std(threshold_6min_severe_rates) if threshold_6min_severe_rates else 0,
                'values': threshold_6min_severe_rates
            },
            'sample_size': len(all_response_times),
            # ★★★ 直近隊選択統計（合計ベースで計算） ★★★
            'closest_dispatch_rate': {
                'mean': (total_closest_dispatches / total_dispatches * 100) if total_dispatches > 0 else 0.0,
                'std': 0.0,  # 合計ベースなので標準偏差は0
                'values': [(total_closest_dispatches / total_dispatches * 100) if total_dispatches > 0 else 0.0]
            },
            'closest_dispatch_rate_by_severity': {
                'severe': {
                    'mean': (total_closest_severe / total_dispatches_severe * 100) if total_dispatches_severe > 0 else 0.0,
                    'std': 0.0,
                    'values': [(total_closest_severe / total_dispatches_severe * 100) if total_dispatches_severe > 0 else 0.0]
                },
                'mild': {
                    'mean': (total_closest_mild / total_dispatches_mild * 100) if total_dispatches_mild > 0 else 0.0,
                    'std': 0.0,
                    'values': [(total_closest_mild / total_dispatches_mild * 100) if total_dispatches_mild > 0 else 0.0]
                },
                'other': {
                    'mean': (total_closest_other / total_dispatches_other * 100) if total_dispatches_other > 0 else 0.0,
                    'std': 0.0,
                    'values': [(total_closest_other / total_dispatches_other * 100) if total_dispatches_other > 0 else 0.0]
                }
            }
        }
        
        # 進捗表示
        print(f"  平均応答時間: {aggregated[strategy]['response_time_overall']['mean']:.2f} ± "
              f"{aggregated[strategy]['response_time_overall']['std']:.2f} 分")
        print(f"  サンプル数: {len(all_response_times)}")
    
    # 集約結果をJSON保存
    summary_path = os.path.join(output_dir, 'aggregated_results.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        # valuesは保存しない（サイズが大きいため）
        json.dump({
            k: {key: val for key, val in v.items() if key != 'values'}
            for k, v in aggregated.items()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n集約結果を保存: {summary_path}")
    
    return aggregated

def visualize_convergence(results: Dict[str, List], output_dir: str):
    """
    実行回数と結果の収束を可視化
    100回の実行が統計的に妥当かを確認
    """
    strategies = EXPERIMENT_CONFIG['strategies']
    strategy_labels = EXPERIMENT_CONFIG['strategy_labels']
    strategy_colors = EXPERIMENT_CONFIG['strategy_colors']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('実行回数による結果の収束', fontsize=16)
    
    # 1. 累積平均の収束プロット
    ax = axes[0, 0]
    for strategy in strategies:
        reports = results[strategy]
        if not reports:
            continue
        means = [r.get('response_times', {}).get('overall', {}).get('mean', 0) for r in reports]
        means = [m for m in means if m > 0]  # 0を除外
        if means:
            cumulative_means = np.cumsum(means) / np.arange(1, len(means) + 1)
            
            ax.plot(range(1, len(cumulative_means) + 1), cumulative_means,
                   label=strategy_labels[strategy], color=strategy_colors[strategy],
                   linewidth=2)
    
    ax.set_xlabel('実行回数')
    ax.set_ylabel('累積平均応答時間（分）')
    ax.set_title('累積平均の収束')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 実行ごとのばらつき
    ax = axes[0, 1]
    for strategy in strategies:
        reports = results[strategy]
        if not reports:
            continue
        means = [r.get('response_times', {}).get('overall', {}).get('mean', 0) for r in reports]
        means = [m for m in means if m > 0]
        if means:
            ax.scatter(range(1, len(means) + 1), means,
                      label=strategy_labels[strategy], 
                      color=strategy_colors[strategy],
                      alpha=0.5, s=20)
    
    ax.set_xlabel('実行回数')
    ax.set_ylabel('平均応答時間（分）')
    ax.set_title('実行ごとのばらつき')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 標準誤差の推移
    ax = axes[1, 0]
    for strategy in strategies:
        reports = results[strategy]
        if not reports:
            continue
        means = [r.get('response_times', {}).get('overall', {}).get('mean', 0) for r in reports]
        means = [m for m in means if m > 0]
        
        if len(means) > 1:
            # 各時点での標準誤差を計算
            sems = [stats.sem(means[:i+1]) for i in range(len(means))]
            
            ax.plot(range(1, len(sems) + 1), sems,
                   label=strategy_labels[strategy], color=strategy_colors[strategy],
                   linewidth=2)
    
    ax.set_xlabel('実行回数')
    ax.set_ylabel('標準誤差（分）')
    ax.set_title('標準誤差の減少')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 信頼区間の幅
    ax = axes[1, 1]
    for strategy in strategies:
        reports = results[strategy]
        if not reports:
            continue
        means = [r.get('response_times', {}).get('overall', {}).get('mean', 0) for r in reports]
        means = [m for m in means if m > 0]
        
        if len(means) > 1:
            # 各時点での95%信頼区間の幅
            ci_widths = []
            for i in range(1, len(means) + 1):
                if i > 1:
                    ci = stats.t.interval(0.95, i-1, loc=np.mean(means[:i]), 
                                         scale=stats.sem(means[:i]))
                    ci_widths.append(ci[1] - ci[0])
                else:
                    ci_widths.append(0)
            
            ax.plot(range(1, len(ci_widths) + 1), ci_widths,
                   label=strategy_labels[strategy], color=strategy_colors[strategy],
                   linewidth=2)
    
    ax.set_xlabel('実行回数')
    ax.set_ylabel('95%信頼区間の幅（分）')
    ax.set_title('推定精度の向上')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"収束分析グラフを保存: {os.path.join(output_dir, 'convergence_analysis.png')}")

def visualize_comparison(analysis: Dict, output_dir: str):
    """比較結果の可視化"""
    strategies = EXPERIMENT_CONFIG['strategies']
    strategy_labels = EXPERIMENT_CONFIG['strategy_labels']
    strategy_colors = EXPERIMENT_CONFIG['strategy_colors']
    
    # 戦略数に応じてレイアウトを調整
    num_strategies = len(strategies)
    
    if num_strategies <= 3:
        # 3つ以下の場合：2行3列のレイアウト（基本6個のグラフ）
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        max_plots = 6  # 最大6個のサブプロット
    else:
        # 4つ以上の場合：3行3列のレイアウト
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()
        max_plots = 9  # 最大9個のサブプロット
    
    fig.suptitle(f'ディスパッチ戦略比較: {len(strategies)}戦略', fontsize=16)
    
    x_pos = np.arange(len(strategies))
    
    # 1. 全体平均応答時間
    ax = axes[0]
    means = [analysis[s]['response_time_overall']['mean'] for s in strategies]
    stds = [analysis[s]['response_time_overall']['std'] for s in strategies]
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                   color=[strategy_colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_ylabel('平均応答時間（分）')
    ax.set_title('全体平均応答時間')
    ax.grid(True, alpha=0.3)
    
    # 数値表示
    # for i, (mean, std) in enumerate(zip(means, stds)):
    #     ax.text(i, mean + std + 0.1, f'{mean:.2f}±{std:.2f}', 
    #             ha='center', va='bottom', fontsize=10)
    
    # 2. 重症系の平均応答時間
    ax = axes[1]
    means_severe = [analysis[s]['response_time_severe']['mean'] for s in strategies]
    stds_severe = [analysis[s]['response_time_severe']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_severe, yerr=stds_severe, capsize=5,
                   color=[strategy_colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_ylabel('平均応答時間（分）')
    ax.set_title('重症・重篤・死亡の平均応答時間')
    ax.grid(True, alpha=0.3)
    
    # 3. 軽症系の平均応答時間
    ax = axes[2]
    means_mild = [analysis[s]['response_time_mild']['mean'] for s in strategies]
    stds_mild = [analysis[s]['response_time_mild']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_mild, yerr=stds_mild, capsize=5,
                   color=[strategy_colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_ylabel('平均応答時間（分）')
    ax.set_title('軽症・中等症の平均応答時間')
    ax.grid(True, alpha=0.3)
    
    # 4. 6分以内達成率（全体）
    ax = axes[3]
    means_6min = [analysis[s]['threshold_6min']['mean'] for s in strategies]
    stds_6min = [analysis[s]['threshold_6min']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_6min, yerr=stds_6min, capsize=5,
                   color=[strategy_colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_ylabel('達成率（%）')
    ax.set_title('6分以内達成率（全体）')
    ax.set_ylim(0, 100)
    #ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='目標90%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 13分以内達成率（全体）
    ax = axes[4]
    means_13min = [analysis[s]['threshold_13min']['mean'] for s in strategies]
    stds_13min = [analysis[s]['threshold_13min']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_13min, yerr=stds_13min, capsize=5,
                   color=[strategy_colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_ylabel('達成率（%）')
    ax.set_title('13分以内達成率（全体）')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # 6. 重症系の6分以内達成率
    ax = axes[5]
    means_6min_severe = [analysis[s]['threshold_6min_severe']['mean'] for s in strategies]
    stds_6min_severe = [analysis[s]['threshold_6min_severe']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_6min_severe, yerr=stds_6min_severe, capsize=5,
                   color=[strategy_colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_ylabel('達成率（%）')
    ax.set_title('6分以内達成率（重症・重篤）')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # 7. 統計的有意性のヒートマップ（4つ以上の戦略がある場合）
    if num_strategies >= 4 and 6 < max_plots:
        ax = axes[6]
        create_significance_heatmap(analysis, strategies, strategy_labels, ax)
    
    # 8. 改善率の比較（ベースライン戦略との比較）
    if num_strategies >= 2:
        # 3つの戦略の場合：6番目のサブプロットに改善率比較を表示
        # 4つ以上の戦略の場合：7番目のサブプロットに改善率比較を表示
        plot_index = 6 if num_strategies == 3 else (7 if 7 < max_plots else 6)
        if plot_index < max_plots:
            ax = axes[plot_index]
            create_improvement_comparison(analysis, strategies, strategy_labels, strategy_colors, ax)
    
    # 未使用のサブプロットを非表示
    for i in range(len(axes)):
        if i >= max_plots:
            axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n比較グラフを保存: {os.path.join(output_dir, 'strategy_comparison.png')}")

def create_significance_heatmap(analysis: Dict, strategies: List[str], strategy_labels: Dict, ax):
    """統計的有意性のヒートマップを作成"""
    num_strategies = len(strategies)
    p_values = np.zeros((num_strategies, num_strategies))
    
    # 重症系応答時間での統計的有意性を計算
    for i, strategy1 in enumerate(strategies):
        for j, strategy2 in enumerate(strategies):
            if i == j:
                p_values[i, j] = 1.0
            else:
                values1 = analysis[strategy1]['response_time_severe']['values']
                values2 = analysis[strategy2]['response_time_severe']['values']
                if len(values1) > 1 and len(values2) > 1:
                    _, p_val = stats.ttest_ind(values1, values2)
                    p_values[i, j] = p_val
                else:
                    p_values[i, j] = 1.0
    
    # ヒートマップの作成
    im = ax.imshow(p_values, cmap='RdYlBu_r', vmin=0, vmax=0.1)
    ax.set_xticks(range(num_strategies))
    ax.set_yticks(range(num_strategies))
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_yticklabels([strategy_labels[s] for s in strategies])
    ax.set_title('統計的有意性（重症系応答時間）\np値 < 0.05で有意差あり')
    
    # 数値表示
    for i in range(num_strategies):
        for j in range(num_strategies):
            text = ax.text(j, i, f'{p_values[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label='p値')

def create_improvement_comparison(analysis: Dict, strategies: List[str], strategy_labels: Dict, strategy_colors: Dict, ax):
    """ベースライン戦略との改善率比較"""
    baseline = strategies[0]  # 最初の戦略をベースラインとする
    improvements = []
    labels = []
    
    for strategy in strategies[1:]:
        baseline_mean = analysis[baseline]['response_time_severe']['mean']
        strategy_mean = analysis[strategy]['response_time_severe']['mean']
        
        if baseline_mean > 0:
            improvement = (baseline_mean - strategy_mean) / baseline_mean * 100
            improvements.append(improvement)
            labels.append(strategy_labels[strategy])
    
    if improvements:
        bars = ax.bar(range(len(improvements)), improvements, 
                     color=[strategy_colors[s] for s in strategies[1:]])
        ax.set_xticks(range(len(improvements)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('改善率（%）')
        ax.set_title(f'ベースライン（{strategy_labels[baseline]}）との比較')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # 数値表示
        for i, (bar, improvement) in enumerate(zip(bars, improvements)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{improvement:+.1f}%', ha='center', va='bottom')

def create_summary_report(analysis: Dict, output_dir: str):
    """サマリーレポートの作成"""
    report_path = os.path.join(output_dir, 'comparison_summary.txt')
    strategies = EXPERIMENT_CONFIG['strategies']
    strategy_labels = EXPERIMENT_CONFIG['strategy_labels']
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ディスパッチ戦略比較実験 サマリーレポート\n")
        f.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"比較戦略数: {len(strategies)}\n")
        f.write("=" * 60 + "\n\n")
        
        # 戦略別の結果
        for strategy in strategies:
            strategy_label = strategy_labels[strategy]
            f.write(f"【{strategy_label}】\n")
            f.write("-" * 40 + "\n")
            
            data = analysis[strategy]
            
            f.write(f"1. 平均応答時間\n")
            f.write(f"   全体: {data['response_time_overall']['mean']:.2f} ± {data['response_time_overall']['std']:.2f} 分\n")
            f.write(f"   重症系: {data['response_time_severe']['mean']:.2f} ± {data['response_time_severe']['std']:.2f} 分\n")
            f.write(f"   軽症系: {data['response_time_mild']['mean']:.2f} ± {data['response_time_mild']['std']:.2f} 分\n\n")
            
            f.write(f"2. 閾値達成率\n")
            f.write(f"   6分以内（全体）: {data['threshold_6min']['mean']:.1f} ± {data['threshold_6min']['std']:.1f} %\n")
            f.write(f"   13分以内（全体）: {data['threshold_13min']['mean']:.1f} ± {data['threshold_13min']['std']:.1f} %\n")
            f.write(f"   6分以内（重症系）: {data['threshold_6min_severe']['mean']:.1f} ± {data['threshold_6min_severe']['std']:.1f} %\n\n")
        
        # 統計的比較結果
        f.write("=" * 60 + "\n")
        f.write("【統計的比較結果】\n")
        f.write("-" * 40 + "\n")
        
        if len(strategies) >= 3:
            # 3つ以上の戦略：ANOVA
            f.write("3群以上の比較（ANOVA）:\n")
            severe_times = [analysis[s]['response_time_severe']['values'] for s in strategies]
            severe_times = [times for times in severe_times if len(times) > 1]
            
            if len(severe_times) >= 3:
                f_stat, p_value = stats.f_oneway(*severe_times)
                f.write(f"  重症系応答時間: F={f_stat:.3f}, p={p_value:.4f}\n")
                f.write(f"  結果: {'有意差あり' if p_value < 0.05 else '有意差なし'} (α=0.05)\n\n")
        
        # ペアワイズ比較
        f.write("ペアワイズ比較（t検定）:\n")
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies[i+1:], i+1):
                values1 = analysis[strategy1]['response_time_severe']['values']
                values2 = analysis[strategy2]['response_time_severe']['values']
                
                if len(values1) > 1 and len(values2) > 1:
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    f.write(f"  {strategy_labels[strategy1]} vs {strategy_labels[strategy2]}: ")
                    f.write(f"t={t_stat:.3f}, p={p_value:.4f} ")
                    f.write(f"({'有意差あり' if p_value < 0.05 else '有意差なし'})\n")
        
        # 改善率の比較
        f.write("\n改善率の比較（ベースライン: 直近隊運用）:\n")
        baseline = strategies[0]
        baseline_mean = analysis[baseline]['response_time_severe']['mean']
        
        for strategy in strategies[1:]:
            strategy_mean = analysis[strategy]['response_time_severe']['mean']
            if baseline_mean > 0:
                improvement = (baseline_mean - strategy_mean) / baseline_mean * 100
                f.write(f"  {strategy_labels[strategy]}: {improvement:+.1f}%\n")
    
    print(f"サマリーレポートを保存: {report_path}")

def create_detailed_summary_report(aggregated: Dict, 
                                   output_dir: str,
                                   experiment_metadata: Dict):
    """
    詳細サマリーレポート（実験設定込み、軽量モード用）
    
    Args:
        aggregated: 集約された統計情報
        output_dir: 出力ディレクトリ
        experiment_metadata: 実験メタデータ
    """
    report_path = os.path.join(output_dir, 'experiment_summary.txt')
    strategies = EXPERIMENT_CONFIG['strategies']
    strategy_labels = EXPERIMENT_CONFIG['strategy_labels']
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # ヘッダー
        f.write("=" * 80 + "\n")
        f.write("救急ディスパッチ戦略比較実験 - 詳細レポート\n")
        f.write("=" * 80 + "\n")
        f.write(f"実験名: {experiment_metadata['experiment_name']}\n")
        f.write(f"作成日時: {experiment_metadata['timestamp']}\n\n")
        
        # 実験設定
        config = experiment_metadata['configuration']
        f.write("【実験設定】\n")
        f.write("-" * 80 + "\n")
        f.write(f"対象期間: {config['start_date']} - {config['end_date']} ({config['total_days']}日間)\n")
        f.write(f"エピソード長: {config['episode_duration_hours']}時間\n")
        f.write(f"実行回数: 各戦略 {config['num_runs']}回（ランダムサンプリング）\n")
        f.write(f"比較戦略数: {len(config['strategies'])}\n")
        f.write(f"戦略一覧: {', '.join([strategy_labels[s] for s in config['strategies']])}\n\n")
        
        # 実験方法
        f.write("【実験方法】\n")
        f.write("-" * 80 + "\n")
        f.write(f"1. {config['total_days']}日間の期間から、各実行ごとにランダムに日付を選択\n")
        f.write(f"2. 選択された日付から{config['episode_duration_hours']}時間のエピソードをシミュレート\n")
        f.write(f"3. 各戦略について{config['num_runs']}回ずつ実行\n")
        f.write(f"4. 結果を集約し、統計的分析を実施\n\n")
        
        # 戦略別結果
        f.write("【戦略別結果】\n")
        f.write("=" * 80 + "\n\n")
        
        for strategy in strategies:
            strategy_label = strategy_labels[strategy]
            f.write(f"■ {strategy_label}\n")
            f.write("-" * 80 + "\n")
            
            data = aggregated[strategy]
            
            f.write(f"1. 平均応答時間\n")
            f.write(f"   全体:\n")
            f.write(f"     平均: {data['response_time_overall']['mean']:.2f} 分\n")
            f.write(f"     標準偏差: {data['response_time_overall']['std']:.2f} 分\n")
            f.write(f"     中央値: {data['response_time_overall']['median']:.2f} 分\n")
            f.write(f"     範囲: {data['response_time_overall']['min']:.2f} - {data['response_time_overall']['max']:.2f} 分\n")
            if data['response_time_overall'].get('ci_95'):
                ci = data['response_time_overall']['ci_95']
                f.write(f"     95%信頼区間: [{ci[0]:.2f}, {ci[1]:.2f}] 分\n")
            
            f.write(f"\n   重症系（重症・重篤・死亡）:\n")
            f.write(f"     平均: {data['response_time_severe']['mean']:.2f} ± {data['response_time_severe']['std']:.2f} 分\n")
            f.write(f"     中央値: {data['response_time_severe']['median']:.2f} 分\n")
            
            f.write(f"\n   軽症系（軽症・中等症）:\n")
            f.write(f"     平均: {data['response_time_mild']['mean']:.2f} ± {data['response_time_mild']['std']:.2f} 分\n")
            f.write(f"     中央値: {data['response_time_mild']['median']:.2f} 分\n\n")
            
            f.write(f"2. 閾値達成率\n")
            f.write(f"   6分以内（全体）: {data['threshold_6min']['mean']:.1f} ± {data['threshold_6min']['std']:.1f} %\n")
            f.write(f"   13分以内（全体）: {data['threshold_13min']['mean']:.1f} ± {data['threshold_13min']['std']:.1f} %\n")
            f.write(f"   6分以内（重症系）: {data['threshold_6min_severe']['mean']:.1f} ± {data['threshold_6min_severe']['std']:.1f} %\n\n")
            
            # ★★★ 直近隊選択率の表示（合計ベース） ★★★
            f.write(f"3. 直近隊選択率\n")
            if 'closest_dispatch_rate' in data and data['closest_dispatch_rate']['mean'] > 0:
                # 合計ベースなので標準偏差は表示しない
                f.write(f"   全体: {data['closest_dispatch_rate']['mean']:.1f} %\n")
                
                # 傷病度別統計
                if 'closest_dispatch_rate_by_severity' in data:
                    by_sev = data['closest_dispatch_rate_by_severity']
                    if by_sev.get('severe', {}).get('mean', 0) > 0:
                        f.write(f"   重症系: {by_sev['severe']['mean']:.1f} %\n")
                    if by_sev.get('mild', {}).get('mean', 0) > 0:
                        f.write(f"   軽症系: {by_sev['mild']['mean']:.1f} %\n")
                    if by_sev.get('other', {}).get('mean', 0) > 0:
                        f.write(f"   その他: {by_sev['other']['mean']:.1f} %\n")
            else:
                f.write(f"   データなし\n")
            f.write("\n")
            
            f.write(f"4. サンプルサイズ\n")
            f.write(f"   実行回数: {data['sample_size']}\n\n")
        
        # 統計的比較
        f.write("【統計的比較】\n")
        f.write("=" * 80 + "\n\n")
        
        if len(strategies) >= 3:
            f.write("■ 多群比較（ANOVA）\n")
            f.write("-" * 80 + "\n")
            f.write("全戦略の平均値が等しいかを検定\n\n")
            
            severe_times = [aggregated[s]['response_time_severe']['values'] for s in strategies]
            severe_times = [times for times in severe_times if len(times) > 1]
            
            if len(severe_times) >= 3:
                f_stat, p_value = stats.f_oneway(*severe_times)
                f.write(f"重症系応答時間のANOVA:\n")
                f.write(f"  F統計量: {f_stat:.3f}\n")
                f.write(f"  p値: {p_value:.4f}\n")
                f.write(f"  結果: {'有意差あり (p < 0.05)' if p_value < 0.05 else '有意差なし (p >= 0.05)'}\n\n")
        
        f.write("■ ペアワイズ比較（t検定）\n")
        f.write("-" * 80 + "\n")
        f.write("各戦略ペアの差が統計的に有意かを検定\n\n")
        
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies[i+1:], i+1):
                values1 = aggregated[strategy1]['response_time_severe']['values']
                values2 = aggregated[strategy2]['response_time_severe']['values']
                
                if len(values1) > 1 and len(values2) > 1:
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    
                    mean_diff = np.mean(values1) - np.mean(values2)
                    
                    f.write(f"{strategy_labels[strategy1]} vs {strategy_labels[strategy2]}:\n")
                    f.write(f"  平均差: {mean_diff:+.2f} 分\n")
                    f.write(f"  t統計量: {t_stat:.3f}\n")
                    f.write(f"  p値: {p_value:.4f}\n")
                    f.write(f"  結果: {'有意差あり (p < 0.05)' if p_value < 0.05 else '有意差なし (p >= 0.05)'}\n\n")
        
        # 改善率分析
        f.write("【改善率分析】\n")
        f.write("=" * 80 + "\n")
        f.write(f"ベースライン戦略: {strategy_labels[strategies[0]]}\n")
        f.write("-" * 80 + "\n\n")
        
        baseline = strategies[0]
        baseline_mean_overall = aggregated[baseline]['response_time_overall']['mean']
        baseline_mean_severe = aggregated[baseline]['response_time_severe']['mean']
        
        for strategy in strategies[1:]:
            f.write(f"■ {strategy_labels[strategy]}\n")
            
            strategy_mean_overall = aggregated[strategy]['response_time_overall']['mean']
            strategy_mean_severe = aggregated[strategy]['response_time_severe']['mean']
            
            improvement_overall = (baseline_mean_overall - strategy_mean_overall) / baseline_mean_overall * 100
            improvement_severe = (baseline_mean_severe - strategy_mean_severe) / baseline_mean_severe * 100
            
            f.write(f"  全体応答時間の改善率: {improvement_overall:+.2f}%\n")
            f.write(f"  重症系応答時間の改善率: {improvement_severe:+.2f}%\n")
            
            # 実際の時間短縮
            time_saved_overall = baseline_mean_overall - strategy_mean_overall
            time_saved_severe = baseline_mean_severe - strategy_mean_severe
            
            f.write(f"  全体応答時間の短縮: {time_saved_overall:+.2f} 分\n")
            f.write(f"  重症系応答時間の短縮: {time_saved_severe:+.2f} 分\n\n")
        
        # 推奨戦略
        f.write("【推奨戦略】\n")
        f.write("=" * 80 + "\n")
        
        # 重症系応答時間が最も短い戦略
        best_strategy_severe = min(strategies, 
                                  key=lambda s: aggregated[s]['response_time_severe']['mean'])
        
        # 全体応答時間が最も短い戦略
        best_strategy_overall = min(strategies, 
                                   key=lambda s: aggregated[s]['response_time_overall']['mean'])
        
        f.write(f"重症系応答時間が最短: {strategy_labels[best_strategy_severe]}\n")
        f.write(f"  平均: {aggregated[best_strategy_severe]['response_time_severe']['mean']:.2f} 分\n\n")
        
        f.write(f"全体応答時間が最短: {strategy_labels[best_strategy_overall]}\n")
        f.write(f"  平均: {aggregated[best_strategy_overall]['response_time_overall']['mean']:.2f} 分\n\n")
        
        # フッター
        f.write("=" * 80 + "\n")
        f.write("レポート終了\n")
        f.write("=" * 80 + "\n")
    
    print(f"詳細サマリーレポートを保存: {report_path}")


if __name__ == "__main__":
    # ============================================================
    # 実験期間リスト（一括実行用）
    # ============================================================
    EXPERIMENT_PERIODS = [
        ("20240204", "20240210"),  # 2月第1週
        ("20240331", "20240406"),  # 3月末〜4月第1週
        ("20240505", "20240511"),  # 5月第1週（GW明け）
        ("20240630", "20240706"),  # 6月末〜7月第1週
        ("20240721", "20240727"),  # 7月第4週（夏季）
        ("20241222", "20241228"),  # 12月第4週（年末）
    ]
    
    # 共通パラメータ
    BASE_PARAMS = {
        'episode_duration_hours': 24,
        'num_runs': 100,
        'output_base_dir': None,
        'wandb_project': 'ems-dispatch-optimization',
        'experiment_name': None
    }
    
    # ============================================================
    # 全期間を順次実行
    # ============================================================
    all_results = {}
    
    for i, (start_date, end_date) in enumerate(EXPERIMENT_PERIODS, 1):
        print("\n" + "=" * 80)
        print(f"【{i}/{len(EXPERIMENT_PERIODS)}】期間: {start_date} - {end_date}")
        print("=" * 80)
        
        try:
            aggregated, experiment_dir = run_comparison_experiment(
                start_date=start_date,
                end_date=end_date,
                episode_duration_hours=BASE_PARAMS['episode_duration_hours'],
                num_runs=BASE_PARAMS['num_runs'],
                output_base_dir=BASE_PARAMS['output_base_dir'],
                wandb_project=BASE_PARAMS['wandb_project'],
                experiment_name=BASE_PARAMS['experiment_name']
            )
            
            all_results[f"{start_date}_{end_date}"] = {
                'aggregated': aggregated,
                'experiment_dir': str(experiment_dir),
                'status': 'success'
            }
            
            print(f"\n✓ 期間 {start_date}-{end_date} 完了: {experiment_dir}")
            
        except Exception as e:
            print(f"\n✗ 期間 {start_date}-{end_date} でエラー発生: {e}")
            all_results[f"{start_date}_{end_date}"] = {
                'status': 'failed',
                'error': str(e)
            }
            continue  # 次の期間へ進む
    
    # ============================================================
    # 全期間の実行サマリー
    # ============================================================
    print("\n" + "=" * 80)
    print("全期間実行サマリー")
    print("=" * 80)
    
    for period, result in all_results.items():
        status = "✓ 成功" if result['status'] == 'success' else "✗ 失敗"
        print(f"  {period}: {status}")
        if result['status'] == 'success':
            print(f"    → {result['experiment_dir']}")
    
    print("=" * 80)