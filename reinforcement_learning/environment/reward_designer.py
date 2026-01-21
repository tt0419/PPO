"""
reward_designer.py
簡素化された報酬設計システム

PPO環境再設計に基づく統一報酬関数
- 重症系: 指数/べき乗型報酬（論文5章の設計）
- 軽症系: 時間報酬 + カバレッジ報酬の重み付け合計
"""

import numpy as np
from typing import Dict
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from constants import is_severe_condition


class RewardDesigner:
    """
    簡素化された報酬設計クラス
    
    モード:
    - 通常PPO: 重症系・軽症系ともにPPO学習対象
    - ハイブリッドPPO: 重症系は直近隊運用（報酬なし）、軽症系のみPPO学習
    """
    
    def __init__(self, config: Dict):
        """
        報酬設計の初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        
        # ===== 報酬パラメータの読み込み =====
        reward_config = config.get('reward', {}).get('unified', {})
        
        # 重症系パラメータ（通常PPO用、論文5章の設計）
        self.critical_params = {
            'target': 6.0,  # 目標時間（分）
            'max_bonus': reward_config.get('critical_max_bonus', 50.0),
            'lambda_param': reward_config.get('critical_lambda', 0.115),
            'penalty_scale': reward_config.get('critical_penalty_scale', 5.0),
            'penalty_power': reward_config.get('critical_penalty_power', 1.5),
        }
        
        # 軽症系パラメータ（論文5章の設計）
        self.mild_params = {
            'target': 13.0,  # 目標時間（分）
            'max_bonus': reward_config.get('mild_max_bonus', 10.0),
            'penalty_scale': reward_config.get('mild_penalty_scale', 1.0),
        }
        
        # カバレッジパラメータ（傷病度考慮運用のロジックに基づく）
        self.coverage_params = {
            'w6': reward_config.get('coverage_w6', 0.5),
            'w13': reward_config.get('coverage_w13', 0.5),
            'penalty_scale': reward_config.get('coverage_penalty_scale', 10.0),
        }
        
        # 重み配分（傷病度考慮運用と同じ: time=0.6, coverage=0.4）
        self.time_weight = reward_config.get('time_weight', 0.6)
        self.coverage_weight = reward_config.get('coverage_weight', 0.4)
        
        # モード設定
        self.hybrid_mode = config.get('hybrid_mode', {}).get('enabled', False)
        
        # システムレベル設定
        system_config = config.get('reward', {}).get('system', {})
        self.dispatch_failure_penalty = system_config.get('dispatch_failure', -1.0)
        self.no_available_penalty = system_config.get('no_available_ambulance', 0.0)
        
        # 初期化完了ログ
        print(f"RewardDesigner初期化完了（簡素化版）:")
        print(f"  ハイブリッドモード: {'有効' if self.hybrid_mode else '無効'}")
        print(f"  重み配分: time={self.time_weight}, coverage={self.coverage_weight}")
        print(f"  カバレッジ配分: w6={self.coverage_params['w6']}, w13={self.coverage_params['w13']}")
    
    def calculate_step_reward(self, severity: str, response_time_sec: float,
                              L6: float = 0.0, L13: float = 0.0) -> float:
        """
        ステップ報酬を計算
        
        Args:
            severity: 傷病度（'重症', '重篤', '死亡', '中等症', '軽症'）
            response_time_sec: 応答時間（秒）
            L6: 6分カバレッジ損失 (0-1)、状態エンコーダーから取得
            L13: 13分カバレッジ損失 (0-1)、状態エンコーダーから取得
        
        Returns:
            報酬値（-100 ~ 100にクリップ）
        """
        rt_min = response_time_sec / 60.0
        
        # ========== 重症系 ==========
        if is_severe_condition(severity):
            if self.hybrid_mode:
                # ハイブリッドモード: 直近隊運用、学習対象外
                return 0.0
            else:
                # 通常PPO: 論文どおりの報酬計算
                return self._calculate_critical_reward(rt_min)
        
        # ========== 軽症系 ==========
        # 時間報酬（論文どおり線形）
        time_reward = self._calculate_mild_time_reward(rt_min)
        
        # カバレッジ報酬（行動レベル、傷病度考慮運用のロジック）
        coverage_reward = self._calculate_coverage_reward(L6, L13)
        
        # 重み付け合計
        total_reward = (self.time_weight * time_reward + 
                       self.coverage_weight * coverage_reward)
        
        return np.clip(total_reward, -100.0, 100.0)
    
    def _calculate_critical_reward(self, rt_min: float) -> float:
        """
        重症系の報酬計算（通常PPO用）
        
        論文5章の設計:
        - 目標時間内: r = B_c × exp(-λt)
        - 目標超過時: r = -α_c × (t - T_c)^ν
        """
        p = self.critical_params
        
        if rt_min <= p['target']:
            # 指数関数的な報酬（早ければ早いほど高い報酬）
            return p['max_bonus'] * np.exp(-p['lambda_param'] * rt_min)
        else:
            # べき乗ペナルティ（超過するほど急激にペナルティ増加）
            overtime = rt_min - p['target']
            return -p['penalty_scale'] * (overtime ** p['penalty_power'])
    
    def _calculate_mild_time_reward(self, rt_min: float) -> float:
        """
        軽症系の時間報酬計算
        
        論文5章の設計:
        - 目標時間内: r = B_m × (1 - t/T_m)
        - 目標超過時: r = -α_m × (t - T_m)
        """
        p = self.mild_params
        
        if rt_min <= p['target']:
            # 線形報酬（目標時間に対する達成度）
            return p['max_bonus'] * (1 - rt_min / p['target'])
        else:
            # 線形ペナルティ
            overtime = rt_min - p['target']
            return -p['penalty_scale'] * overtime
    
    def _calculate_coverage_reward(self, L6: float, L13: float) -> float:
        """
        カバレッジ報酬計算（行動レベル）
        
        傷病度考慮運用のロジック:
        - 選んだ隊が出場することによるカバレッジ損失をペナルティとして計算
        - 損失が小さい = 良い選択 → 報酬が高い（ペナルティが小さい）
        - 損失が大きい = 悪い選択 → 報酬が低い（ペナルティが大きい）
        """
        p = self.coverage_params
        
        # カバレッジ損失の重み付け合計
        combined_loss = p['w6'] * L6 + p['w13'] * L13
        
        # 損失をペナルティに変換（損失が大きいほどペナルティが大きい）
        return -p['penalty_scale'] * combined_loss
    
    def get_failure_penalty(self, failure_type: str) -> float:
        """
        失敗ペナルティを取得
        
        Args:
            failure_type: 'dispatch', 'no_available', 'unhandled'
        """
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
            'time_weight': self.time_weight,
            'coverage_weight': self.coverage_weight,
            'critical_params': self.critical_params,
            'mild_params': self.mild_params,
            'coverage_params': self.coverage_params,
        }
    
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
    
    # ===== 後方互換性のためのプロパティ =====
    @property
    def mode(self) -> str:
        """後方互換性: modeプロパティ"""
        return 'unified'
    
    def update_curriculum(self, episode: int):
        """後方互換性: カリキュラム学習は無効化"""
        pass
