"""
reward_designer.py v10修正案
傷病度考慮運用と完全同一のスコアベース報酬モードを追加

【修正箇所】
1. __init__: score_based_mode パラメータの読み込み
2. calculate_step_reward: スコアベースモード分岐の追加
3. _calculate_score_based_reward: 新規メソッド追加
"""

import numpy as np
from typing import Dict
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from constants import is_severe_condition


class RewardDesigner:
    """
    簡素化された報酬設計クラス（v10: スコアベースモード対応）
    """
    
    def __init__(self, config: Dict):
        """
        報酬設計の初期化
        """
        self.config = config
        
        # ===== 報酬パラメータの読み込み =====
        reward_config = config.get('reward', {}).get('unified', {})
        
        # ★★★ v10新機能: スコアベースモード ★★★
        self.score_based_mode = reward_config.get('score_based_mode', False)
        self.score_scale = reward_config.get('score_scale', 10.0)
        self.time_score_weight = reward_config.get('time_score_weight', 0.6)
        self.coverage_loss_weight = reward_config.get('coverage_loss_weight', 0.4)
        self.time_normalization = reward_config.get('time_normalization', 13.0)
        
        # 重症系パラメータ（通常PPO用、論文5章の設計）
        self.critical_params = {
            'target': 6.0,
            'max_bonus': reward_config.get('critical_max_bonus', 50.0),
            'lambda_param': reward_config.get('critical_lambda', 0.115),
            'penalty_scale': reward_config.get('critical_penalty_scale', 5.0),
            'penalty_power': reward_config.get('critical_penalty_power', 1.5),
        }
        
        # 軽症系パラメータ（従来モード用）
        self.mild_params = {
            'target': 13.0,
            'max_bonus': reward_config.get('mild_max_bonus', 10.0),
            'penalty_scale': reward_config.get('mild_penalty_scale', 1.0),
        }
        
        # カバレッジパラメータ
        self.coverage_params = {
            'w6': reward_config.get('coverage_w6', 0.5),
            'w13': reward_config.get('coverage_w13', 0.5),
            'penalty_scale': reward_config.get('coverage_penalty_scale', 10.0),
            'penalty_max': reward_config.get('coverage_penalty_max', float('inf')),
        }
        
        # 重み配分（従来モード用）
        self.time_weight = reward_config.get('time_weight', 0.6)
        self.coverage_weight = reward_config.get('coverage_weight', 0.4)
        
        # モード設定
        self.hybrid_mode = config.get('hybrid_mode', {}).get('enabled', False)
        
        # システムレベル設定
        system_config = config.get('reward', {}).get('system', {})
        self.dispatch_failure_penalty = system_config.get('dispatch_failure', -1.0)
        self.no_available_penalty = system_config.get('no_available_ambulance', 0.0)
        
        # 初期化完了ログ
        print(f"RewardDesigner初期化完了:")
        print(f"  ハイブリッドモード: {'有効' if self.hybrid_mode else '無効'}")
        print(f"  ★★★ スコアベースモード: {'有効' if self.score_based_mode else '無効'} ★★★")
        if self.score_based_mode:
            print(f"  スコア計算: {self.time_score_weight}*time + {self.coverage_loss_weight}*coverage")
            print(f"  時間正規化: {self.time_normalization}分")
            print(f"  報酬スケール: {self.score_scale}")
        else:
            print(f"  重み配分: time={self.time_weight}, coverage={self.coverage_weight}")
            print(f"  カバレッジペナルティ上限: {self.coverage_params['penalty_max']}")
    
    def calculate_step_reward(self, severity: str, response_time_sec: float,
                              L6: float = 0.0, L13: float = 0.0) -> float:
        """
        ステップ報酬を計算
        """
        rt_min = response_time_sec / 60.0
        
        # ========== 重症系 ==========
        if is_severe_condition(severity):
            if self.hybrid_mode:
                return 0.0
            else:
                return self._calculate_critical_reward(rt_min)
        
        # ========== 軽症系 ==========
        # ★★★ v10: スコアベースモード分岐 ★★★
        if self.score_based_mode:
            return self._calculate_score_based_reward(rt_min, L6, L13)
        else:
            # 従来モード
            time_reward = self._calculate_mild_time_reward(rt_min)
            coverage_reward = self._calculate_coverage_reward(L6, L13)
            total_reward = (self.time_weight * time_reward + 
                           self.coverage_weight * coverage_reward)
            return np.clip(total_reward, -100.0, 100.0)
    
    def _calculate_score_based_reward(self, rt_min: float, L6: float, L13: float) -> float:
        """
        ★★★ v10新機能: 傷病度考慮運用と完全同一のスコアベース報酬 ★★★
        
        傷病度考慮運用（dispatch_strategies.py）の計算式:
        - time_score = travel_time / time_threshold_13min
        - coverage_loss = 0.5 * loss_6min + 0.5 * loss_13min
        - combined_score = 0.6 * time_score + 0.4 * coverage_loss
        - → score最小の救急隊を選択
        
        PPO報酬への変換:
        - reward = -score_scale * combined_score
        - → reward最大化 = score最小化
        """
        # 時間スコア（傷病度考慮運用と同一）
        time_score = rt_min / self.time_normalization  # 13分で1.0
        
        # カバレッジ損失（傷病度考慮運用と同一）
        w6 = self.coverage_params['w6']
        w13 = self.coverage_params['w13']
        coverage_loss = w6 * L6 + w13 * L13
        
        # 複合スコア（傷病度考慮運用と同一）
        combined_score = (self.time_score_weight * time_score + 
                         self.coverage_loss_weight * coverage_loss)
        
        # スコアを反転して報酬に（score最小 → reward最大）
        reward = -self.score_scale * combined_score
        
        return reward
    
    def _calculate_critical_reward(self, rt_min: float) -> float:
        """重症系の報酬計算（通常PPO用）"""
        p = self.critical_params
        
        if rt_min <= p['target']:
            return p['max_bonus'] * np.exp(-p['lambda_param'] * rt_min)
        else:
            overtime = rt_min - p['target']
            return -p['penalty_scale'] * (overtime ** p['penalty_power'])
    
    def _calculate_mild_time_reward(self, rt_min: float) -> float:
        """軽症系の時間報酬計算（従来モード用）"""
        p = self.mild_params
        
        if rt_min <= p['target']:
            return p['max_bonus'] * (1 - rt_min / p['target'])
        else:
            overtime = rt_min - p['target']
            return -p['penalty_scale'] * overtime
    
    def _calculate_coverage_reward(self, L6: float, L13: float) -> float:
        """カバレッジ報酬計算（従来モード用）"""
        p = self.coverage_params
        
        combined_loss = p['w6'] * L6 + p['w13'] * L13
        raw_penalty = p['penalty_scale'] * combined_loss
        penalty_max = p.get('penalty_max', float('inf'))
        capped_penalty = min(raw_penalty, penalty_max)
        
        return -capped_penalty
    
    def get_failure_penalty(self, failure_type: str) -> float:
        """失敗ペナルティを取得"""
        if failure_type == 'dispatch':
            return self.dispatch_failure_penalty
        elif failure_type == 'no_available':
            return self.no_available_penalty
        else:
            return -10.0
    
    def get_info(self) -> Dict:
        """現在の報酬設定情報を返す"""
        return {
            'hybrid_mode': self.hybrid_mode,
            'score_based_mode': self.score_based_mode,
            'time_weight': self.time_weight,
            'coverage_weight': self.coverage_weight,
            'critical_params': self.critical_params,
            'mild_params': self.mild_params,
            'coverage_params': self.coverage_params,
        }
    
    def calculate_episode_reward(self, stats: Dict) -> float:
        """後方互換性: エピソード報酬計算"""
        if stats.get('total_dispatches', 0) == 0:
            return 0.0
        
        total = stats['total_dispatches']
        response_times = stats.get('response_times', [])
        if not response_times:
            return 0.0
        
        avg_rt = np.mean(response_times)
        rate_13min = stats.get('achieved_13min', 0) / total
        rate_6min = stats.get('achieved_6min', 0) / total
        
        critical_total = stats.get('critical_total', 0)
        if critical_total > 0:
            critical_6min_rate = stats.get('critical_6min', 0) / critical_total
        else:
            critical_6min_rate = 1.0
        
        episode_reward = (
            rate_13min * 50 +
            critical_6min_rate * 30 -
            avg_rt
        )
        
        return episode_reward
    
    @property
    def mode(self) -> str:
        """後方互換性: modeプロパティ"""
        return 'score_based' if self.score_based_mode else 'unified'
    
    def update_curriculum(self, episode: int):
        """後方互換性: カリキュラム学習は無効化"""
        pass


# ================================================================
# 期待される効果
# ================================================================
# 
# 【傷病度考慮運用のスコア計算】
# time_score = rt / 13  → 範囲: 0〜1+（13分超過で1超）
# coverage_loss = 0.5*L6 + 0.5*L13  → 範囲: 0〜1
# combined_score = 0.6*time + 0.4*coverage  → 範囲: 0〜1+
# → score最小の救急隊を選択
# 
# 【v10のPPO報酬】
# reward = -10 * combined_score  → 範囲: -10〜0（13分以内）
# → reward最大化 = score最小化
# 
# 【期待される効果】
# - PPOが傷病度考慮運用と同じ評価基準で学習
# - 直近隊選択率 40-60%に近づく可能性
# 
# ================================================================
