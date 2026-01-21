"""
generate_dashboard.py
実験結果CSVを読み込み、インタラクティブなHTMLダッシュボードを生成

使い方:
    python generate_dashboard.py [csv_path] [output_path]
    
    csv_path: 入力CSVファイルのパス（デフォルト: all_experiment_results.csv）
    output_path: 出力HTMLファイルのパス（デフォルト: experiment_dashboard.html）
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """CSVを読み込んでデータをクリーニング"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # 新旧フォーマット統一
    if '重症RT' in df.columns:
        df['severe_rt'] = df['重症RT']
        df['severe_rt_std'] = df['重症RT_std']
        df['overall_rt'] = df['全体RT']
        df['overall_rt_std'] = df['全体RT_std']
        df['rate_6min_severe'] = df['6分率_重症']
        df['rate_6min_severe_std'] = df['6分率_重症_std']
        df['rate_13min'] = df['13分率_全体']
        df['rate_13min_std'] = df['13分率_全体_std']
        df['closest_rate'] = df['直近隊率_全体']
        df['closest_rate_severe'] = df['直近隊率_重症']
        df['closest_rate_mild'] = df['直近隊率_軽症']
        df['test_start'] = df['テスト開始日']
        df['strategy_name'] = df['戦略表示名']
    
    # 旧フォーマットのフォールバック
    if 'severe_rt' not in df.columns or df['severe_rt'].isna().all():
        df['severe_rt'] = df.get('response_time_severe_mean', np.nan)
        df['severe_rt_std'] = df.get('response_time_severe_std', np.nan)
        df['overall_rt'] = df.get('response_time_overall_mean', np.nan)
        df['overall_rt_std'] = df.get('response_time_overall_std', np.nan)
        df['rate_6min_severe'] = df.get('threshold_6min_severe_mean', np.nan)
        df['rate_6min_severe_std'] = df.get('threshold_6min_severe_std', np.nan)
        df['rate_13min'] = df.get('threshold_13min_mean', np.nan)
        df['rate_13min_std'] = df.get('threshold_13min_std', np.nan)
        df['test_start'] = df.get('start_date', np.nan)
        df['strategy_name'] = df.get('strategy', df.get('戦略ID', 'Unknown'))
    
    # 有効なデータのみ抽出
    df = df[df['severe_rt'].notna()].copy()
    
    # テスト開始日を文字列に変換
    df['test_start'] = df['test_start'].astype(str).str.replace('.0', '', regex=False)
    
    # 季節・象限がない場合は推定
    if '季節' not in df.columns:
        df['季節'] = df['test_start'].apply(lambda x: get_season(str(x)))
    if '象限' not in df.columns:
        df['象限'] = '不明'
    
    return df


def get_season(date_str: str) -> str:
    """日付から季節を判定"""
    try:
        month = int(date_str[4:6])
        if month in [3, 4, 5]:
            return "春"
        elif month in [6, 7, 8]:
            return "夏"
        elif month in [9, 10, 11]:
            return "秋"
        else:
            return "冬"
    except:
        return "不明"


def prepare_heatmap_data(df: pd.DataFrame, metric: str) -> dict:
    """ヒートマップ用のデータを準備"""
    # テスト期間と戦略のピボットテーブル
    pivot = df.pivot_table(
        index='strategy_name',
        columns='test_start',
        values=metric,
        aggfunc='mean'
    )
    
    # 期間を日付順にソート
    sorted_cols = sorted(pivot.columns, key=lambda x: str(x))
    pivot = pivot[sorted_cols]
    
    return {
        'strategies': pivot.index.tolist(),
        'periods': [str(p) for p in pivot.columns.tolist()],
        'values': pivot.values.tolist(),
        'metric': metric
    }


def prepare_bar_data(df: pd.DataFrame) -> dict:
    """棒グラフ用のデータを準備"""
    periods = sorted(df['test_start'].unique(), key=lambda x: str(x))
    strategies = df['strategy_name'].unique().tolist()
    
    data_by_period = {}
    for period in periods:
        period_df = df[df['test_start'] == period]
        data_by_period[str(period)] = {
            'strategies': period_df['strategy_name'].tolist(),
            'severe_rt': period_df['severe_rt'].tolist(),
            'severe_rt_std': period_df['severe_rt_std'].fillna(0).tolist(),
            'overall_rt': period_df['overall_rt'].tolist(),
            'rate_6min_severe': period_df['rate_6min_severe'].tolist(),
            'rate_13min': period_df['rate_13min'].tolist(),
            'closest_rate': period_df['closest_rate'].fillna(0).tolist(),
            'season': period_df['季節'].iloc[0] if len(period_df) > 0 else '不明',
            'quadrant': period_df['象限'].iloc[0] if len(period_df) > 0 else '不明'
        }
    
    return {
        'periods': [str(p) for p in periods],
        'all_strategies': strategies,
        'by_period': data_by_period
    }


def prepare_ppo_data(df: pd.DataFrame) -> dict:
    """PPOパラメータ分析用のデータを準備"""
    ppo_df = df[df['strategy_name'].str.contains('PPO', na=False)].copy()
    
    if len(ppo_df) == 0:
        return {'models': [], 'data': []}
    
    # モデルごとにグループ化
    models = ppo_df['strategy_name'].unique().tolist()
    
    model_data = []
    for model in models:
        model_df = ppo_df[ppo_df['strategy_name'] == model]
        
        # パラメータを取得（最初の行から）
        first_row = model_df.iloc[0]
        params = {
            'hybrid_mode': first_row.get('hybrid_mode', 'N/A'),
            'time_weight': first_row.get('time_weight', 'N/A'),
            'coverage_weight': first_row.get('coverage_weight', 'N/A'),
            'coverage_penalty_scale': first_row.get('coverage_penalty_scale', 'N/A'),
            'entropy_coef': first_row.get('entropy_coef', 'N/A'),
        }
        
        # 各期間の結果
        results = []
        for _, row in model_df.iterrows():
            results.append({
                'period': str(row['test_start']),
                'severe_rt': row['severe_rt'],
                'rate_13min': row['rate_13min'],
                'closest_rate': row.get('closest_rate', 0)
            })
        
        model_data.append({
            'name': model,
            'params': params,
            'results': results
        })
    
    return {
        'models': models,
        'data': model_data
    }


def prepare_table_data(df: pd.DataFrame) -> list:
    """詳細テーブル用のデータを準備"""
    table_data = []
    for _, row in df.iterrows():
        table_data.append({
            'period': str(row['test_start']),
            'season': row.get('季節', '不明'),
            'quadrant': row.get('象限', '不明'),
            'strategy': row['strategy_name'],
            'severe_rt': f"{row['severe_rt']:.2f}" if pd.notna(row['severe_rt']) else 'N/A',
            'severe_rt_std': f"{row['severe_rt_std']:.2f}" if pd.notna(row.get('severe_rt_std')) else 'N/A',
            'overall_rt': f"{row['overall_rt']:.2f}" if pd.notna(row['overall_rt']) else 'N/A',
            'rate_6min': f"{row['rate_6min_severe']:.1f}" if pd.notna(row['rate_6min_severe']) else 'N/A',
            'rate_13min': f"{row['rate_13min']:.1f}" if pd.notna(row['rate_13min']) else 'N/A',
            'closest_rate': f"{row['closest_rate']:.1f}" if pd.notna(row.get('closest_rate')) else 'N/A',
            'closest_rate_severe': f"{row['closest_rate_severe']:.1f}" if pd.notna(row.get('closest_rate_severe')) else 'N/A',
            'closest_rate_mild': f"{row['closest_rate_mild']:.1f}" if pd.notna(row.get('closest_rate_mild')) else 'N/A',
        })
    
    return table_data


def generate_html(df: pd.DataFrame, output_path: str):
    """HTMLダッシュボードを生成"""
    
    # データ準備
    heatmap_severe = prepare_heatmap_data(df, 'severe_rt')
    heatmap_overall = prepare_heatmap_data(df, 'overall_rt')
    heatmap_13min = prepare_heatmap_data(df, 'rate_13min')
    heatmap_6min = prepare_heatmap_data(df, 'rate_6min_severe')
    bar_data = prepare_bar_data(df)
    ppo_data = prepare_ppo_data(df)
    table_data = prepare_table_data(df)
    
    # 最良戦略の算出
    best_by_period = {}
    for period in bar_data['periods']:
        period_data = bar_data['by_period'][period]
        if period_data['severe_rt']:
            min_idx = np.argmin(period_data['severe_rt'])
            best_by_period[period] = {
                'strategy': period_data['strategies'][min_idx],
                'severe_rt': period_data['severe_rt'][min_idx]
            }
    
    html_content = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EMS実験結果ダッシュボード</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-yellow: #eab308;
            --accent-purple: #a855f7;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', 'Meiryo', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }}
        
        .header {{
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
            padding: 24px 32px;
            border-bottom: 1px solid var(--bg-tertiary);
        }}
        
        .header h1 {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        
        .header .subtitle {{
            color: var(--text-secondary);
            font-size: 14px;
        }}
        
        .tabs {{
            display: flex;
            gap: 4px;
            padding: 16px 32px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--bg-tertiary);
        }}
        
        .tab {{
            padding: 12px 24px;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.2s;
        }}
        
        .tab:hover {{
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }}
        
        .tab.active {{
            background: var(--accent-blue);
            color: white;
        }}
        
        .content {{
            padding: 32px;
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        .tab-panel {{
            display: none;
        }}
        
        .tab-panel.active {{
            display: block;
        }}
        
        .card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        
        .card-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .card-title .badge {{
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 4px;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }}
        
        .grid {{
            display: grid;
            gap: 24px;
        }}
        
        .grid-2 {{
            grid-template-columns: repeat(2, 1fr);
        }}
        
        .grid-3 {{
            grid-template-columns: repeat(3, 1fr);
        }}
        
        @media (max-width: 1200px) {{
            .grid-2, .grid-3 {{
                grid-template-columns: 1fr;
            }}
        }}
        
        /* ヒートマップ */
        .heatmap-container {{
            overflow-x: auto;
        }}
        
        .heatmap {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        
        .heatmap th, .heatmap td {{
            padding: 12px 16px;
            text-align: center;
            border: 1px solid var(--bg-tertiary);
        }}
        
        .heatmap th {{
            background: var(--bg-tertiary);
            font-weight: 600;
            white-space: nowrap;
        }}
        
        .heatmap th.strategy-col {{
            text-align: left;
            max-width: 200px;
        }}
        
        .heatmap td.strategy-cell {{
            text-align: left;
            font-weight: 500;
            white-space: nowrap;
        }}
        
        .heatmap td.value-cell {{
            font-weight: 600;
            transition: transform 0.2s;
        }}
        
        .heatmap td.value-cell:hover {{
            transform: scale(1.05);
        }}
        
        .heatmap td.best {{
            box-shadow: inset 0 0 0 2px var(--accent-green);
        }}
        
        /* 概要カード */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .summary-card {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 20px;
        }}
        
        .summary-card .label {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        
        .summary-card .value {{
            font-size: 28px;
            font-weight: 700;
        }}
        
        .summary-card .detail {{
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 4px;
        }}
        
        /* チャート */
        .chart-container {{
            position: relative;
            height: 400px;
        }}
        
        /* 期間セレクター */
        .period-selector {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        
        .period-btn {{
            padding: 8px 16px;
            background: var(--bg-tertiary);
            border: none;
            border-radius: 6px;
            color: var(--text-secondary);
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .period-btn:hover {{
            background: var(--accent-blue);
            color: white;
        }}
        
        .period-btn.active {{
            background: var(--accent-blue);
            color: white;
        }}
        
        .period-info {{
            display: flex;
            gap: 16px;
            margin-bottom: 16px;
        }}
        
        .period-info .tag {{
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 13px;
            font-weight: 500;
        }}
        
        .period-info .season {{
            background: var(--accent-purple);
            color: white;
        }}
        
        .period-info .quadrant {{
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }}
        
        /* テーブル */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        
        .data-table th, .data-table td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--bg-tertiary);
        }}
        
        .data-table th {{
            background: var(--bg-tertiary);
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        
        .data-table tr:hover {{
            background: var(--bg-tertiary);
        }}
        
        .data-table .num {{
            text-align: right;
            font-family: 'Consolas', monospace;
        }}
        
        /* PPOカード */
        .ppo-card {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
        }}
        
        .ppo-card .model-name {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--accent-blue);
        }}
        
        .ppo-params {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }}
        
        .ppo-param {{
            background: var(--bg-secondary);
            padding: 8px 12px;
            border-radius: 4px;
        }}
        
        .ppo-param .param-name {{
            font-size: 11px;
            color: var(--text-secondary);
            margin-bottom: 2px;
        }}
        
        .ppo-param .param-value {{
            font-size: 14px;
            font-weight: 600;
        }}
        
        .ppo-results {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 8px;
        }}
        
        .ppo-result {{
            background: var(--bg-secondary);
            padding: 12px;
            border-radius: 4px;
            text-align: center;
        }}
        
        .ppo-result .period {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }}
        
        .ppo-result .rt {{
            font-size: 18px;
            font-weight: 700;
        }}
        
        /* フィルター */
        .filters {{
            display: flex;
            gap: 16px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .filter-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .filter-group label {{
            font-size: 13px;
            color: var(--text-secondary);
        }}
        
        .filter-group select {{
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 13px;
        }}
        
        /* ランキング */
        .ranking {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .ranking-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
        }}
        
        .ranking-item .rank {{
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--accent-blue);
            color: white;
            border-radius: 50%;
            font-size: 13px;
            font-weight: 600;
        }}
        
        .ranking-item .rank.gold {{
            background: linear-gradient(135deg, #fbbf24, #f59e0b);
        }}
        
        .ranking-item .rank.silver {{
            background: linear-gradient(135deg, #9ca3af, #6b7280);
        }}
        
        .ranking-item .rank.bronze {{
            background: linear-gradient(135deg, #d97706, #b45309);
        }}
        
        .ranking-item .info {{
            flex: 1;
        }}
        
        .ranking-item .strategy-name {{
            font-weight: 600;
            margin-bottom: 2px;
        }}
        
        .ranking-item .period {{
            font-size: 12px;
            color: var(--text-secondary);
        }}
        
        .ranking-item .value {{
            font-size: 18px;
            font-weight: 700;
            color: var(--accent-green);
        }}
        
        /* 生成時刻 */
        .footer {{
            text-align: center;
            padding: 24px;
            color: var(--text-secondary);
            font-size: 12px;
        }}
        
        /* 戦略フィルター（チェックボックス） */
        .strategy-filter {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
        }}
        
        .strategy-filter-title {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .strategy-filter-title button {{
            font-size: 11px;
            padding: 4px 8px;
            background: var(--bg-secondary);
            border: none;
            border-radius: 4px;
            color: var(--text-secondary);
            cursor: pointer;
            margin-left: 8px;
        }}
        
        .strategy-filter-title button:hover {{
            background: var(--accent-blue);
            color: white;
        }}
        
        .strategy-checkboxes {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        
        .strategy-checkbox {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: var(--bg-secondary);
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 13px;
        }}
        
        .strategy-checkbox:hover {{
            background: var(--accent-blue);
            color: white;
        }}
        
        .strategy-checkbox input {{
            cursor: pointer;
        }}
        
        .strategy-checkbox.unchecked {{
            opacity: 0.5;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚑 EMS配車戦略 実験結果ダッシュボード</h1>
        <div class="subtitle">生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | データ件数: {len(df)}件</div>
    </div>
    
    <div class="tabs">
        <button class="tab active" onclick="showTab('overview')">📊 概要</button>
        <button class="tab" onclick="showTab('comparison')">📈 期間別比較</button>
        <button class="tab" onclick="showTab('heatmap')">🗺️ ヒートマップ</button>
        <button class="tab" onclick="showTab('ppo')">🤖 PPO分析</button>
        <button class="tab" onclick="showTab('table')">📋 詳細データ</button>
    </div>
    
    <div class="content">
        <!-- グローバル戦略フィルター -->
        <div class="strategy-filter">
            <div class="strategy-filter-title">
                🎯 表示する戦略を選択
                <button onclick="selectAllStrategies()">すべて選択</button>
                <button onclick="deselectAllStrategies()">すべて解除</button>
                <button onclick="selectPPOOnly()">PPOのみ</button>
                <button onclick="selectNonPPOOnly()">非PPOのみ</button>
            </div>
            <div class="strategy-checkboxes" id="strategyCheckboxes"></div>
        </div>
        
        <!-- 概要タブ -->
        <div id="overview" class="tab-panel active">
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="label">テスト期間数</div>
                    <div class="value">{len(bar_data['periods'])}</div>
                    <div class="detail">{', '.join([p[:4]+'/'+p[4:6]+'/'+p[6:] for p in bar_data['periods'][:3]])}...</div>
                </div>
                <div class="summary-card">
                    <div class="label">比較戦略数</div>
                    <div class="value">{len(bar_data['all_strategies'])}</div>
                    <div class="detail">PPO: {len([s for s in bar_data['all_strategies'] if 'PPO' in str(s)])}種類</div>
                </div>
                <div class="summary-card">
                    <div class="label">最良重症RT</div>
                    <div class="value" style="color: var(--accent-green);">{min([v['severe_rt'] for v in best_by_period.values()]):.2f}分</div>
                    <div class="detail">期間・戦略により変動</div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">🏆 期間別ベスト戦略</div>
                <div class="ranking" id="rankingContainer">
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">📊 戦略別 平均重症RT</div>
                <div class="chart-container">
                    <canvas id="overviewChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- 期間別比較タブ -->
        <div id="comparison" class="tab-panel">
            <div class="card">
                <div class="card-title">テスト期間を選択</div>
                <div class="period-selector" id="periodSelector">
                </div>
                <div id="periodInfo" class="period-info">
                    <span class="tag season">{bar_data['by_period'][bar_data['periods'][0]]['season']}</span>
                    <span class="tag quadrant">{bar_data['by_period'][bar_data['periods'][0]]['quadrant']}</span>
                </div>
            </div>
            
            <div class="grid grid-2">
                <div class="card">
                    <div class="card-title">重症系 応答時間（分）<span class="badge">低いほど良い</span></div>
                    <div class="chart-container">
                        <canvas id="severeRtChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">13分以内達成率（%）<span class="badge">高いほど良い</span></div>
                    <div class="chart-container">
                        <canvas id="rate13minChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="grid grid-2">
                <div class="card">
                    <div class="card-title">6分以内達成率 - 重症（%）<span class="badge">高いほど良い</span></div>
                    <div class="chart-container">
                        <canvas id="rate6minChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">直近隊選択率（%）</div>
                    <div class="chart-container">
                        <canvas id="closestRateChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- ヒートマップタブ -->
        <div id="heatmap" class="tab-panel">
            <div class="card">
                <div class="card-title">① 重症系 応答時間（分）ヒートマップ <span class="badge">低いほど良い（緑）</span></div>
                <div class="heatmap-container">
                    <table class="heatmap" id="heatmapSevere"></table>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">② 全体 応答時間（分）ヒートマップ <span class="badge">低いほど良い（緑）</span></div>
                <div class="heatmap-container">
                    <table class="heatmap" id="heatmapOverall"></table>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">③ 6分以内達成率 - 重症（%）ヒートマップ <span class="badge">高いほど良い（緑）</span></div>
                <div class="heatmap-container">
                    <table class="heatmap" id="heatmap6min"></table>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">④ 13分以内達成率（%）ヒートマップ <span class="badge">高いほど良い（緑）</span></div>
                <div class="heatmap-container">
                    <table class="heatmap" id="heatmap13min"></table>
                </div>
            </div>
        </div>
        
        <!-- PPO分析タブ -->
        <div id="ppo" class="tab-panel">
            <div class="card">
                <div class="card-title">🤖 PPOモデル パラメータと結果</div>
                <div id="ppoCards"></div>
            </div>
            
            <div class="card">
                <div class="card-title">PPOモデル比較チャート</div>
                <div class="chart-container">
                    <canvas id="ppoCompareChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- 詳細データタブ -->
        <div id="table" class="tab-panel">
            <div class="card">
                <div class="card-title">📋 全実験結果</div>
                <div class="filters">
                    <div class="filter-group">
                        <label>期間:</label>
                        <select id="filterPeriod" onchange="filterTable()">
                            <option value="">すべて</option>
                        </select>
                    </div>
                    <div class="filter-group">
                        <label>戦略:</label>
                        <select id="filterStrategy" onchange="filterTable()">
                            <option value="">すべて</option>
                        </select>
                    </div>
                </div>
                <div style="overflow-x: auto;">
                    <table class="data-table" id="dataTable">
                        <thead>
                            <tr>
                                <th>期間</th>
                                <th>季節</th>
                                <th>象限</th>
                                <th>戦略</th>
                                <th class="num">重症RT</th>
                                <th class="num">±std</th>
                                <th class="num">全体RT</th>
                                <th class="num">6分率重症</th>
                                <th class="num">13分率</th>
                                <th class="num">直近隊率全体</th>
                                <th class="num">直近隊率重症</th>
                                <th class="num">直近隊率軽症</th>
                            </tr>
                        </thead>
                        <tbody id="dataTableBody"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        EMS配車戦略 実験結果ダッシュボード | Generated by generate_dashboard.py
    </div>
    
    <script>
        // データ
        const barData = {json.dumps(bar_data, ensure_ascii=False)};
        const heatmapSevere = {json.dumps(heatmap_severe, ensure_ascii=False)};
        const heatmapOverall = {json.dumps(heatmap_overall, ensure_ascii=False)};
        const heatmap13min = {json.dumps(heatmap_13min, ensure_ascii=False)};
        const heatmap6min = {json.dumps(heatmap_6min, ensure_ascii=False)};
        const ppoData = {json.dumps(ppo_data, ensure_ascii=False)};
        const tableData = {json.dumps(table_data, ensure_ascii=False)};
        const bestByPeriod = {json.dumps(best_by_period, ensure_ascii=False)};
        
        // 現在選択中の期間
        let currentPeriod = barData.periods[0];
        
        // 選択中の戦略（フィルター用）
        let selectedStrategies = new Set(barData.all_strategies);
        
        // チャートインスタンス
        let severeRtChart, rate13minChart, rate6minChart, closestRateChart, overviewChart, ppoCompareChart;
        
        // カラーパレット
        const colors = [
            '#3b82f6', '#22c55e', '#ef4444', '#eab308', '#a855f7',
            '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16'
        ];
        
        // 戦略フィルター初期化
        function initStrategyFilter() {{
            const container = document.getElementById('strategyCheckboxes');
            container.innerHTML = barData.all_strategies.map(s => `
                <label class="strategy-checkbox">
                    <input type="checkbox" value="${{s}}" checked onchange="onStrategyFilterChange()">
                    ${{s}}
                </label>
            `).join('');
        }}
        
        // 戦略フィルター変更時
        function onStrategyFilterChange() {{
            selectedStrategies = new Set(
                Array.from(document.querySelectorAll('#strategyCheckboxes input:checked')).map(cb => cb.value)
            );
            
            // チェックボックスの見た目を更新
            document.querySelectorAll('.strategy-checkbox').forEach(label => {{
                const input = label.querySelector('input');
                label.classList.toggle('unchecked', !input.checked);
            }});
            
            // 各表示を更新
            updateAllDisplays();
        }}
        
        // すべて選択
        function selectAllStrategies() {{
            document.querySelectorAll('#strategyCheckboxes input').forEach(cb => cb.checked = true);
            onStrategyFilterChange();
        }}
        
        // すべて解除
        function deselectAllStrategies() {{
            document.querySelectorAll('#strategyCheckboxes input').forEach(cb => cb.checked = false);
            onStrategyFilterChange();
        }}
        
        // PPOのみ選択
        function selectPPOOnly() {{
            document.querySelectorAll('#strategyCheckboxes input').forEach(cb => {{
                cb.checked = cb.value.includes('PPO');
            }});
            onStrategyFilterChange();
        }}
        
        // 非PPOのみ選択
        function selectNonPPOOnly() {{
            document.querySelectorAll('#strategyCheckboxes input').forEach(cb => {{
                cb.checked = !cb.value.includes('PPO');
            }});
            onStrategyFilterChange();
        }}
        
        // 全表示を更新
        function updateAllDisplays() {{
            updateCharts();
            updateOverviewChart();
            renderHeatmap('heatmapSevere', heatmapSevere, true);
            renderHeatmap('heatmapOverall', heatmapOverall, true);
            renderHeatmap('heatmap6min', heatmap6min, false);
            renderHeatmap('heatmap13min', heatmap13min, false);
            renderPpoCards();
            renderTable();
            filterTable();
        }}
        
        // 概要チャート更新
        function updateOverviewChart() {{
            // 戦略ごとの平均重症RTを計算（フィルター適用）
            const strategyAvgRt = {{}};
            Object.values(barData.by_period).forEach(period => {{
                period.strategies.forEach((s, i) => {{
                    if (!selectedStrategies.has(s)) return;
                    if (!strategyAvgRt[s]) strategyAvgRt[s] = [];
                    strategyAvgRt[s].push(period.severe_rt[i]);
                }});
            }});
            
            const avgLabels = Object.keys(strategyAvgRt);
            const avgData = avgLabels.map(s => {{
                const vals = strategyAvgRt[s].filter(v => v != null);
                return vals.length ? vals.reduce((a, b) => a + b) / vals.length : 0;
            }});
            
            overviewChart.data.labels = avgLabels;
            overviewChart.data.datasets[0].data = avgData;
            overviewChart.data.datasets[0].backgroundColor = avgLabels.map((_, i) => colors[i % colors.length]);
            overviewChart.update();
        }}
        
        // タブ切り替え
        function showTab(tabId) {{
            document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }}
        
        // 期間選択
        function selectPeriod(period) {{
            currentPeriod = period;
            document.querySelectorAll('.period-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.textContent.replace(/\\//g, '') === period);
            }});
            
            const periodData = barData.by_period[period];
            document.getElementById('periodInfo').innerHTML = `
                <span class="tag season">${{periodData.season}}</span>
                <span class="tag quadrant">${{periodData.quadrant}}</span>
            `;
            
            updateCharts();
        }}
        
        // チャート更新
        function updateCharts() {{
            const periodData = barData.by_period[currentPeriod];
            
            // フィルター適用
            const filteredIndices = periodData.strategies
                .map((s, i) => selectedStrategies.has(s) ? i : -1)
                .filter(i => i >= 0);
            
            const strategies = filteredIndices.map(i => periodData.strategies[i]);
            const bgColors = filteredIndices.map((_, i) => colors[i % colors.length]);
            
            // 重症RTチャート
            severeRtChart.data.labels = strategies;
            severeRtChart.data.datasets[0].data = filteredIndices.map(i => periodData.severe_rt[i]);
            severeRtChart.data.datasets[0].backgroundColor = bgColors;
            severeRtChart.update();
            
            // 13分率チャート
            rate13minChart.data.labels = strategies;
            rate13minChart.data.datasets[0].data = filteredIndices.map(i => periodData.rate_13min[i]);
            rate13minChart.data.datasets[0].backgroundColor = bgColors;
            rate13minChart.update();
            
            // 6分率チャート
            rate6minChart.data.labels = strategies;
            rate6minChart.data.datasets[0].data = filteredIndices.map(i => periodData.rate_6min_severe[i]);
            rate6minChart.data.datasets[0].backgroundColor = bgColors;
            rate6minChart.update();
            
            // 直近隊率チャート
            closestRateChart.data.labels = strategies;
            closestRateChart.data.datasets[0].data = filteredIndices.map(i => periodData.closest_rate[i]);
            closestRateChart.data.datasets[0].backgroundColor = bgColors;
            closestRateChart.update();
        }}
        
        // ヒートマップ描画
        function renderHeatmap(tableId, data, isLowerBetter) {{
            const table = document.getElementById(tableId);
            let html = '<thead><tr><th class="strategy-col">戦略</th>';
            
            data.periods.forEach(p => {{
                html += `<th>${{p.substring(0,4)}}/${{p.substring(4,6)}}/${{p.substring(6)}}</th>`;
            }});
            html += '</tr></thead><tbody>';
            
            // フィルタリングされた戦略のインデックス
            const filteredIndices = data.strategies
                .map((s, i) => selectedStrategies.has(s) ? i : -1)
                .filter(i => i >= 0);
            
            // フィルタリングされたデータで最良値を計算
            const bestInCol = data.periods.map((_, colIdx) => {{
                const colValues = filteredIndices
                    .map(rowIdx => data.values[rowIdx][colIdx])
                    .filter(v => v != null && !isNaN(v));
                if (colValues.length === 0) return null;
                return isLowerBetter ? Math.min(...colValues) : Math.max(...colValues);
            }});
            
            // フィルタリングされた全値（色計算用）
            const allFilteredValues = filteredIndices
                .flatMap(rowIdx => data.values[rowIdx])
                .filter(v => v != null && !isNaN(v));
            
            filteredIndices.forEach(rowIdx => {{
                const strategy = data.strategies[rowIdx];
                html += `<tr><td class="strategy-cell">${{strategy}}</td>`;
                data.values[rowIdx].forEach((val, colIdx) => {{
                    if (val == null || isNaN(val)) {{
                        html += '<td class="value-cell">-</td>';
                    }} else {{
                        const isBest = bestInCol[colIdx] != null && Math.abs(val - bestInCol[colIdx]) < 0.01;
                        const color = getHeatmapColor(val, allFilteredValues, isLowerBetter);
                        html += `<td class="value-cell ${{isBest ? 'best' : ''}}" style="background: ${{color}}">${{val.toFixed(2)}}</td>`;
                    }}
                }});
                html += '</tr>';
            }});
            
            html += '</tbody>';
            table.innerHTML = html;
        }}
        
        function getHeatmapColor(value, allValues, isLowerBetter) {{
            const validValues = allValues.filter(v => v != null && !isNaN(v));
            const min = Math.min(...validValues);
            const max = Math.max(...validValues);
            const range = max - min || 1;
            
            let ratio = (value - min) / range;
            if (isLowerBetter) ratio = 1 - ratio;
            
            // 緑（良い）から赤（悪い）へのグラデーション
            const r = Math.round(255 * (1 - ratio));
            const g = Math.round(180 * ratio);
            const b = Math.round(80 * (1 - ratio * 0.5));
            
            return `rgba(${{r}}, ${{g}}, ${{b}}, 0.6)`;
        }}
        
        // PPOカード描画
        function renderPpoCards() {{
            const container = document.getElementById('ppoCards');
            let html = '';
            
            // フィルター適用
            const filteredModels = ppoData.data.filter(model => selectedStrategies.has(model.name));
            
            filteredModels.forEach(model => {{
                html += `
                <div class="ppo-card">
                    <div class="model-name">${{model.name}}</div>
                    <div class="ppo-params">
                        <div class="ppo-param">
                            <div class="param-name">hybrid_mode</div>
                            <div class="param-value">${{model.params.hybrid_mode}}</div>
                        </div>
                        <div class="ppo-param">
                            <div class="param-name">time_weight</div>
                            <div class="param-value">${{model.params.time_weight}}</div>
                        </div>
                        <div class="ppo-param">
                            <div class="param-name">coverage_weight</div>
                            <div class="param-value">${{model.params.coverage_weight}}</div>
                        </div>
                        <div class="ppo-param">
                            <div class="param-name">coverage_penalty</div>
                            <div class="param-value">${{model.params.coverage_penalty_scale}}</div>
                        </div>
                        <div class="ppo-param">
                            <div class="param-name">entropy_coef</div>
                            <div class="param-value">${{model.params.entropy_coef}}</div>
                        </div>
                    </div>
                    <div class="ppo-results">
                        ${{model.results.map(r => `
                            <div class="ppo-result">
                                <div class="period">${{r.period.substring(0,4)}}/${{r.period.substring(4,6)}}/${{r.period.substring(6)}}</div>
                                <div class="rt" style="color: ${{r.severe_rt < 9 ? 'var(--accent-green)' : r.severe_rt > 12 ? 'var(--accent-red)' : 'var(--text-primary)'}}">${{r.severe_rt.toFixed(2)}}分</div>
                            </div>
                        `).join('')}}
                    </div>
                </div>
                `;
            }});
            
            container.innerHTML = html || '<p>選択されたPPOモデルがありません</p>';
        }}
        
        // テーブル描画
        function renderTable() {{
            const tbody = document.getElementById('dataTableBody');
            let html = '';
            
            tableData.forEach(row => {{
                html += `
                <tr data-period="${{row.period}}" data-strategy="${{row.strategy}}">
                    <td>${{row.period.substring(0,4)}}/${{row.period.substring(4,6)}}/${{row.period.substring(6)}}</td>
                    <td>${{row.season}}</td>
                    <td>${{row.quadrant}}</td>
                    <td>${{row.strategy}}</td>
                    <td class="num">${{row.severe_rt}}</td>
                    <td class="num">${{row.severe_rt_std}}</td>
                    <td class="num">${{row.overall_rt}}</td>
                    <td class="num">${{row.rate_6min}}</td>
                    <td class="num">${{row.rate_13min}}</td>
                    <td class="num">${{row.closest_rate}}</td>
                    <td class="num">${{row.closest_rate_severe}}</td>
                    <td class="num">${{row.closest_rate_mild}}</td>
                </tr>
                `;
            }});
            
            tbody.innerHTML = html;
        }}
        
        // テーブルフィルタ
        function filterTable() {{
            const periodFilter = document.getElementById('filterPeriod').value;
            const strategyFilter = document.getElementById('filterStrategy').value;
            
            document.querySelectorAll('#dataTableBody tr').forEach(row => {{
                const matchPeriod = !periodFilter || row.dataset.period === periodFilter;
                const matchStrategyDropdown = !strategyFilter || row.dataset.strategy === strategyFilter;
                const matchStrategyGlobal = selectedStrategies.has(row.dataset.strategy);
                row.style.display = matchPeriod && matchStrategyDropdown && matchStrategyGlobal ? '' : 'none';
            }});
        }}
        
        // 初期化
        document.addEventListener('DOMContentLoaded', function() {{
            // 戦略フィルター初期化
            initStrategyFilter();
            
            // 動的要素の生成
            
            // 1. ランキング
            const rankingContainer = document.getElementById('rankingContainer');
            const sortedPeriods = Object.keys(bestByPeriod).sort((a, b) => bestByPeriod[a].severe_rt - bestByPeriod[b].severe_rt);
            rankingContainer.innerHTML = sortedPeriods.map((p, i) => `
                <div class="ranking-item">
                    <div class="rank ${{i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : ''}}">${{i + 1}}</div>
                    <div class="info">
                        <div class="strategy-name">${{bestByPeriod[p].strategy}}</div>
                        <div class="period">${{p.substring(0,4)}}/${{p.substring(4,6)}}/${{p.substring(6)}}週 | ${{barData.by_period[p].season}} | ${{barData.by_period[p].quadrant}}</div>
                    </div>
                    <div class="value">${{bestByPeriod[p].severe_rt.toFixed(2)}}分</div>
                </div>
            `).join('');
            
            // 2. 期間セレクター
            const periodSelector = document.getElementById('periodSelector');
            periodSelector.innerHTML = barData.periods.map((p, i) => 
                `<button class="period-btn ${{i === 0 ? 'active' : ''}}" onclick="selectPeriod('${{p}}')">${{p.substring(0,4)}}/${{p.substring(4,6)}}/${{p.substring(6)}}</button>`
            ).join('');
            
            // 3. フィルターオプション
            const filterPeriod = document.getElementById('filterPeriod');
            barData.periods.forEach(p => {{
                filterPeriod.innerHTML += `<option value="${{p}}">${{p.substring(0,4)}}/${{p.substring(4,6)}}/${{p.substring(6)}}</option>`;
            }});
            
            const filterStrategy = document.getElementById('filterStrategy');
            barData.all_strategies.forEach(s => {{
                filterStrategy.innerHTML += `<option value="${{s}}">${{s}}</option>`;
            }});
            
            // 概要チャート
            const ctxOverview = document.getElementById('overviewChart').getContext('2d');
            
            // 戦略ごとの平均重症RTを計算
            const strategyAvgRt = {{}};
            Object.values(barData.by_period).forEach(period => {{
                period.strategies.forEach((s, i) => {{
                    if (!strategyAvgRt[s]) strategyAvgRt[s] = [];
                    strategyAvgRt[s].push(period.severe_rt[i]);
                }});
            }});
            
            const avgLabels = Object.keys(strategyAvgRt);
            const avgData = avgLabels.map(s => {{
                const vals = strategyAvgRt[s].filter(v => v != null);
                return vals.length ? vals.reduce((a, b) => a + b) / vals.length : 0;
            }});
            
            overviewChart = new Chart(ctxOverview, {{
                type: 'bar',
                data: {{
                    labels: avgLabels,
                    datasets: [{{
                        label: '平均重症RT（分）',
                        data: avgData,
                        backgroundColor: avgLabels.map((_, i) => colors[i % colors.length])
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: false,
                            title: {{ display: true, text: '分' }}
                        }}
                    }}
                }}
            }});
            
            // 期間別チャート初期化
            const periodData = barData.by_period[currentPeriod];
            const bgColors = periodData.strategies.map((_, i) => colors[i % colors.length]);
            
            const chartOptions = {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{ y: {{ beginAtZero: false }} }}
            }};
            
            severeRtChart = new Chart(document.getElementById('severeRtChart'), {{
                type: 'bar',
                data: {{
                    labels: periodData.strategies,
                    datasets: [{{ data: periodData.severe_rt, backgroundColor: bgColors }}]
                }},
                options: chartOptions
            }});
            
            rate13minChart = new Chart(document.getElementById('rate13minChart'), {{
                type: 'bar',
                data: {{
                    labels: periodData.strategies,
                    datasets: [{{ data: periodData.rate_13min, backgroundColor: bgColors }}]
                }},
                options: {{ ...chartOptions, scales: {{ y: {{ beginAtZero: true, max: 100 }} }} }}
            }});
            
            rate6minChart = new Chart(document.getElementById('rate6minChart'), {{
                type: 'bar',
                data: {{
                    labels: periodData.strategies,
                    datasets: [{{ data: periodData.rate_6min_severe, backgroundColor: bgColors }}]
                }},
                options: {{ ...chartOptions, scales: {{ y: {{ beginAtZero: true, max: 100 }} }} }}
            }});
            
            closestRateChart = new Chart(document.getElementById('closestRateChart'), {{
                type: 'bar',
                data: {{
                    labels: periodData.strategies,
                    datasets: [{{ data: periodData.closest_rate, backgroundColor: bgColors }}]
                }},
                options: {{ ...chartOptions, scales: {{ y: {{ beginAtZero: true, max: 100 }} }} }}
            }});
            
            // PPO比較チャート
            if (ppoData.data.length > 0) {{
                const ppoLabels = [...new Set(ppoData.data.flatMap(m => m.results.map(r => r.period)))].sort();
                const ppoDatasets = ppoData.data.map((model, i) => ({{
                    label: model.name,
                    data: ppoLabels.map(p => {{
                        const result = model.results.find(r => r.period === p);
                        return result ? result.severe_rt : null;
                    }}),
                    borderColor: colors[i % colors.length],
                    backgroundColor: colors[i % colors.length] + '40',
                    fill: false,
                    tension: 0.3
                }}));
                
                ppoCompareChart = new Chart(document.getElementById('ppoCompareChart'), {{
                    type: 'line',
                    data: {{
                        labels: ppoLabels.map(p => p.substring(4,6) + '/' + p.substring(6)),
                        datasets: ppoDatasets
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ position: 'bottom' }}
                        }},
                        scales: {{
                            y: {{
                                title: {{ display: true, text: '重症RT（分）' }}
                            }}
                        }}
                    }}
                }});
            }}
            
            // ヒートマップ描画（順番: 重症RT → 全体RT → 6分率 → 13分率）
            renderHeatmap('heatmapSevere', heatmapSevere, true);
            renderHeatmap('heatmapOverall', heatmapOverall, true);
            renderHeatmap('heatmap6min', heatmap6min, false);
            renderHeatmap('heatmap13min', heatmap13min, false);
            
            // PPOカード描画
            renderPpoCards();
            
            // テーブル描画
            renderTable();
        }});
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"ダッシュボードを生成しました: {output_path}")


def main():
    # コマンドライン引数
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'all_experiment_results.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'experiment_dashboard.html'
    
    print(f"CSVファイル読み込み: {csv_path}")
    df = load_and_clean_data(csv_path)
    print(f"有効データ: {len(df)}件")
    
    generate_html(df, output_path)


if __name__ == '__main__':
    main()
