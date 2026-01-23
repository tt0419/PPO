# ================================================================
# RewardDesigner 修正案
# カバレッジペナルティに上限を設定
# ================================================================

# ===== 修正箇所1: __init__メソッド内 =====
# 以下を追加（coverage_paramsの定義部分）

# 【変更前】
# self.coverage_params = {
#     'w6': reward_config.get('coverage_w6', 0.5),
#     'w13': reward_config.get('coverage_w13', 0.5),
#     'penalty_scale': reward_config.get('coverage_penalty_scale', 10.0),
# }

# 【変更後】
self.coverage_params = {
    'w6': reward_config.get('coverage_w6', 0.5),
    'w13': reward_config.get('coverage_w13', 0.5),
    'penalty_scale': reward_config.get('coverage_penalty_scale', 10.0),
    'penalty_max': reward_config.get('coverage_penalty_max', float('inf')),  # 新規追加
}

# ===== 修正箇所2: _calculate_coverage_rewardメソッド =====

# 【変更前】
# def _calculate_coverage_reward(self, L6: float, L13: float) -> float:
#     """
#     カバレッジ報酬計算（行動レベル）
#     """
#     p = self.coverage_params
#     combined_loss = p['w6'] * L6 + p['w13'] * L13
#     return -p['penalty_scale'] * combined_loss

# 【変更後】
def _calculate_coverage_reward(self, L6: float, L13: float) -> float:
    """
    カバレッジ報酬計算（行動レベル）
    
    傷病度考慮運用のロジック:
    - 選んだ隊が出場することによるカバレッジ損失をペナルティとして計算
    - 損失が小さい = 良い選択 → 報酬が高い（ペナルティが小さい）
    - 損失が大きい = 悪い選択 → 報酬が低い（ペナルティが大きい）
    
    【v9新機能】ペナルティ上限の設定
    - coverage_penalty_maxが設定されている場合、ペナルティを上限でクリップ
    - これにより、カバレッジ損失の過度な最小化を防ぐ
    """
    p = self.coverage_params
    
    # カバレッジ損失の重み付け合計
    combined_loss = p['w6'] * L6 + p['w13'] * L13
    
    # 損失をペナルティに変換
    raw_penalty = p['penalty_scale'] * combined_loss
    
    # ★★★ 新機能: ペナルティ上限でクリップ ★★★
    penalty_max = p.get('penalty_max', float('inf'))
    capped_penalty = min(raw_penalty, penalty_max)
    
    return -capped_penalty


# ===== 修正箇所3: __init__メソッドの初期化完了ログ =====
# 以下を追加

print(f"RewardDesigner初期化完了（簡素化版）:")
print(f"  ハイブリッドモード: {'有効' if self.hybrid_mode else '無効'}")
print(f"  重み配分: time={self.time_weight}, coverage={self.coverage_weight}")
print(f"  カバレッジ配分: w6={self.coverage_params['w6']}, w13={self.coverage_params['w13']}")
print(f"  カバレッジペナルティ上限: {self.coverage_params['penalty_max']}")  # 新規追加


# ================================================================
# 期待される効果
# ================================================================
# 
# 【現在の問題】
# coverage_penalty = -35 * combined_loss
# combined_loss ∈ [0, 1] なので、penalty ∈ [0, -35]
# PPOは際限なくcombined_lossを最小化しようとする
# → 直近隊選択率10%（カバレッジ過重視）
# 
# 【修正後】
# coverage_penalty_max = 5.0 の場合:
# raw_penalty = 35 * combined_loss
# capped_penalty = min(raw_penalty, 5.0)
# coverage_penalty = -capped_penalty ∈ [0, -5]
# 
# 上限に達したら、それ以上カバレッジを最小化してもペナルティは減らない
# → 時間報酬（0〜10）との勝負になる
# → 直近隊選択率が上昇する見込み
# 
# ================================================================
# パラメータ設定の目安
# ================================================================
# 
# | coverage_penalty_max | 期待される直近隊率 | 備考 |
# |----------------------|-------------------|------|
# | 3.0 | 60-80% | 時間重視寄り |
# | 5.0 | 40-60% | バランス |
# | 7.0 | 20-40% | カバレッジ重視寄り |
# | inf（デフォルト） | 10-20% | 現状と同じ |
# 
