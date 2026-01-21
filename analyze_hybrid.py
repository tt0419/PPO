import pandas as pd
import numpy as np

# --- 設定 ---
# ここに、分析したいwandbのCSVファイル名を設定してください
CSV_FILENAME = "wandb_export_2025-09-14T21_18_33.380+09_00.csv"
# --- 設定ここまで ---

def analyze_results(filename: str):
    """
    WandBのCSVエクスポートを読み込み、性能を分析してサマリーを表示する
    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {filename}")
        return

    # カラム名を扱いやすいように修正
    df.columns = [col.replace('/', '_') for col in df.columns]
    
    # 安定した性能を評価するため、最後の500エピソードを対象とする
    stable_period = df.tail(min(500, len(df)))

    # --- 主要指標の計算 ---
    mean_reward = stable_period['reward_episode'].mean()
    mean_rt = stable_period['performance_mean_response_time'].mean()
    mean_6min_rate = stable_period['performance_6min_achievement_rate'].mean()
    
    # 重症度別の指標
    severe_6min_rate = stable_period.get('severity_severe_6min_rate', pd.Series(np.nan)).mean()
    critical_6min_rate = stable_period.get('severity_critical_6min_rate', pd.Series(np.nan)).mean()
    mild_mean_rt = stable_period.get('severity_mild_mean_time', pd.Series(np.nan)).mean()
    moderate_mean_rt = stable_period.get('severity_moderate_mean_time', pd.Series(np.nan)).mean()
    
    # --- 比較用のベンチマーク ---
    baseline_closest = {
        "Strategy": "ベースライン（直近隊）",
        "平均報酬": -1093.92,
        "平均応答時間 (分)": 8.55,
        "6分達成率 (%)": 28.5
    }
    
    best_full_ppo = {
        "Strategy": "PPO単体（最良）",
        "平均報酬": -1036.75,
        "平均応答時間 (分)": 8.39,
        "6分達成率 (%)": 30.3
    }
    
    hybrid_strategy = {
        "Strategy": "ハイブリッド戦略（今回）",
        "平均報酬": mean_reward,
        "平均応答時間 (分)": mean_rt,
        "6分達成率 (%)": mean_6min_rate * 100
    }
    
    # --- 結果の表示 ---
    comparison_df = pd.DataFrame([baseline_closest, best_full_ppo, hybrid_strategy])
    
    print("### ハイブリッド戦略 性能評価サマリー")
    print("\n--- 最終500エピソードでの平均性能 ---")
    print(comparison_df.round(2).to_markdown(index=False))
    
    print("\n--- 重症度別 詳細（ハイブリッド戦略）---")
    # np.isnanでチェック
    if not np.isnan(severe_6min_rate):
        print(f"  - 重症(Severe) 6分達成率: {severe_6min_rate*100:.2f}%")
    if not np.isnan(critical_6min_rate):
        print(f"  - 重篤(Critical) 6分達成率: {critical_6min_rate*100:.2f}%")
    if not np.isnan(mild_mean_rt):
        print(f"  - 軽症(Mild) 平均応答時間: {mild_mean_rt:.2f}分")
    if not np.isnan(moderate_mean_rt):
        print(f"  - 中等症(Moderate) 平均応答時間: {moderate_mean_rt:.2f}分")

if __name__ == "__main__":
    analyze_results(CSV_FILENAME)