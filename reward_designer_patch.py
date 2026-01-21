"""
reward_designer.py への追加コード

以下のメソッドを reward_designer.py の get_info() メソッドの後に追加してください。
（約180行目付近、update_curriculum メソッドの前）
"""

# ===== 追加するメソッド =====

def calculate_episode_reward(self, stats: Dict) -> float:
    """
    後方互換性: エピソード報酬計算
    
    新設計ではステップ報酬の累積を使用するため、
    このメソッドは応答時間統計から簡易的な報酬を計算
    
    Args:
        stats: エピソード統計辞書
            - total_dispatches: 総配車数
            - response_times: 応答時間リスト（分）
            - achieved_13min: 13分達成数
            - achieved_6min: 6分達成数
            - critical_total: 重症系総数
            - critical_6min: 重症系6分達成数
    
    Returns:
        エピソード報酬（参考値）
    """
    if stats.get('total_dispatches', 0) == 0:
        return 0.0
    
    total = stats['total_dispatches']
    
    # 応答時間ベースの簡易報酬
    response_times = stats.get('response_times', [])
    if not response_times:
        return 0.0
    
    avg_rt = np.mean(response_times)
    
    # 達成率
    rate_13min = stats.get('achieved_13min', 0) / total
    rate_6min = stats.get('achieved_6min', 0) / total
    
    # 重症系6分達成率
    critical_total = stats.get('critical_total', 0)
    if critical_total > 0:
        critical_6min_rate = stats.get('critical_6min', 0) / critical_total
    else:
        critical_6min_rate = 1.0  # 重症系がない場合は満点
    
    # 簡易報酬計算
    # - 13分達成率ボーナス（最大50点）
    # - 重症系6分達成率ボーナス（最大30点）
    # - 平均応答時間ペナルティ（-avg_rt）
    episode_reward = (
        rate_13min * 50 +
        critical_6min_rate * 30 -
        avg_rt
    )
    
    return episode_reward


# ===== 使用方法 =====
# 
# 1. reward_designer.py を開く
# 2. get_info() メソッドを見つける（約170行目）
# 3. その後に上記の calculate_episode_reward メソッドを追加
# 4. クラス内のインデントに合わせる（4スペース）
#
# 例:
#
#     def get_info(self) -> Dict:
#         """現在の報酬設定情報を返す"""
#         return {...}
#     
#     def calculate_episode_reward(self, stats: Dict) -> float:  # ← ここに追加
#         """後方互換性: エピソード報酬計算"""
#         ...
#     
#     @property
#     def mode(self) -> str:
#         ...
