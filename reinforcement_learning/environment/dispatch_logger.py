"""
dispatch_logger.py
救急車配車の詳細ログを管理するモジュール
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

class DispatchLogger:
    """救急車配車の詳細ログを記録・管理するクラス"""
    
    def __init__(self, log_dir: str = "logs/dispatch", enabled: bool = True):
        """
        Args:
            log_dir: ログファイルの保存ディレクトリ
            enabled: ログ機能の有効/無効
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # ログデータの蓄積
        self.dispatch_logs = []
        self.episode_summary = {}
        
        # ログファイル名（タイムスタンプ付き）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"dispatch_log_{timestamp}.csv"
        self.summary_file = self.log_dir / f"episode_summary_{timestamp}.json"
        
        # CSVヘッダーの初期化
        if self.enabled:
            self._init_csv_file()
    
    def _init_csv_file(self):
        """CSVファイルのヘッダーを初期化"""
        headers = [
            'episode', 'step', 'timestamp',
            'call_id', 'call_severity', 'call_location_h3',
            'selected_ambulance_id', 'ambulance_name', 'ambulance_type', 'ambulance_station_h3',
            'response_time_minutes', 'travel_distance_km',
            'available_ambulances_count', 'total_ambulances_count',
            'action_mask_valid_count', 'optimal_ambulance_id',
            'optimal_response_time_minutes', 'teacher_match',
            'reward', 'episode_reward_avg'
        ]
        
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_dispatch(self, 
                    episode: int,
                    step: int,
                    call_info: Dict,
                    selected_ambulance_id: int,
                    ambulance_info: Dict,
                    response_time_minutes: float,
                    available_count: int,
                    total_count: int,
                    action_mask_valid_count: int,
                    optimal_ambulance_id: Optional[int] = None,
                    optimal_response_time: Optional[float] = None,
                    teacher_match: bool = False,
                    reward: float = 0.0,
                    episode_reward_avg: float = 0.0):
        """
        配車ログを記録
        
        Args:
            episode: エピソード番号
            step: ステップ番号
            call_info: 事案情報
            selected_ambulance_id: 選択された救急車ID
            ambulance_info: 救急車情報
            response_time_minutes: 応答時間（分）
            available_count: 利用可能救急車数
            total_count: 総救急車数
            action_mask_valid_count: 有効な行動数
            optimal_ambulance_id: 最適救急車ID
            optimal_response_time: 最適応答時間
            teacher_match: 教師との一致
            reward: 報酬
            episode_reward_avg: エピソード平均報酬
        """
        if not self.enabled:
            return
        
        # 移動距離の推定（簡易版）
        travel_distance = self._estimate_distance(ambulance_info, call_info)
        
        # ログエントリを作成
        log_entry = {
            'episode': episode,
            'step': step,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'call_id': call_info.get('id', 'unknown'),
            'call_severity': call_info.get('severity', 'unknown'),
            'call_location_h3': call_info.get('h3_index', 'unknown'),
            'selected_ambulance_id': selected_ambulance_id,
            'ambulance_name': ambulance_info.get('name', f"救急車{selected_ambulance_id}"),
            'ambulance_type': 'virtual' if ambulance_info.get('is_virtual', False) else 'real',
            'ambulance_station_h3': ambulance_info.get('station_h3', 'unknown'),
            'response_time_minutes': round(response_time_minutes, 2),
            'travel_distance_km': round(travel_distance, 2),
            'available_ambulances_count': available_count,
            'total_ambulances_count': total_count,
            'action_mask_valid_count': action_mask_valid_count,
            'optimal_ambulance_id': optimal_ambulance_id or -1,
            'optimal_response_time_minutes': round(optimal_response_time or 0, 2),
            'teacher_match': teacher_match,
            'reward': round(reward, 2),
            'episode_reward_avg': round(episode_reward_avg, 2)
        }
        
        # メモリに蓄積
        self.dispatch_logs.append(log_entry)
        
        # CSVファイルに即座に書き込み
        self._write_to_csv(log_entry)
        
        # コンソールにも簡潔な情報を出力
        self._print_dispatch_summary(log_entry)
    
    def _estimate_distance(self, ambulance_info: Dict, call_info: Dict) -> float:
        """移動距離を推定（簡易版）"""
        try:
            # 応答時間から距離を推定（平均時速30km/h）
            response_time_hours = ambulance_info.get('response_time_minutes', 10) / 60.0
            return response_time_hours * 30.0
        except:
            return 5.0  # デフォルト5km
    
    def _write_to_csv(self, log_entry: Dict):
        """CSVファイルに書き込み"""
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(list(log_entry.values()))
        except Exception as e:
            print(f"ログ書き込みエラー: {e}")
    
    def _print_dispatch_summary(self, log_entry: Dict):
        """配車サマリーをコンソールに出力"""
        episode = log_entry['episode']
        step = log_entry['step']
        call_severity = log_entry['call_severity']
        ambulance_id = log_entry['selected_ambulance_id']
        ambulance_type = log_entry['ambulance_type']
        response_time = log_entry['response_time_minutes']
        available_count = log_entry['available_ambulances_count']
        
        # 10ステップごとに詳細情報を出力
        if step % 10 == 0:
            # 救急車タイプを日本語で表示
            type_jp = "仮想" if ambulance_type == "virtual" else "実車"
            amb_name = log_entry.get('ambulance_name', f"救急車{ambulance_id}")
            print(f"[配車] Ep{episode}-{step}: {call_severity} → {amb_name}({type_jp}) "
                  f"{response_time:.1f}分 (利用可能:{available_count}台)")
    
    def log_episode_summary(self, episode: int, summary: Dict):
        """エピソードサマリーを記録"""
        if not self.enabled:
            return
        
        self.episode_summary[episode] = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_dispatches': summary.get('total_dispatches', 0),
            'mean_response_time': summary.get('mean_response_time', 0),
            'achieved_6min_rate': summary.get('achieved_6min_rate', 0),
            'virtual_ambulance_usage': summary.get('virtual_ambulance_usage', {}),
            'teacher_match_rate': summary.get('teacher_match_rate', 0),
            'episode_reward': summary.get('episode_reward', 0)
        }
        
        # JSONファイルに保存
        self._save_episode_summary()
    
    def _save_episode_summary(self):
        """エピソードサマリーをJSONファイルに保存"""
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(self.episode_summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"サマリー保存エラー: {e}")
    
    def get_dispatch_statistics(self) -> Dict:
        """配車統計を取得"""
        if not self.dispatch_logs:
            return {}
        
        df = pd.DataFrame(self.dispatch_logs)
        
        stats = {
            'total_dispatches': len(df),
            'virtual_ambulance_usage': df['ambulance_type'].value_counts().to_dict(),
            'severity_distribution': df['call_severity'].value_counts().to_dict(),
            'mean_response_time': df['response_time_minutes'].mean(),
            'teacher_match_rate': df['teacher_match'].mean(),
            'optimal_vs_selected': {
                'optimal_selected_count': len(df[df['selected_ambulance_id'] == df['optimal_ambulance_id']]),
                'total_count': len(df[df['optimal_ambulance_id'] != -1])
            }
        }
        
        return stats
    
    def close(self):
        """ログファイルを閉じる"""
        if self.enabled and self.episode_summary:
            self._save_episode_summary()
            print(f"配車ログを保存しました: {self.log_file}")
            print(f"エピソードサマリーを保存しました: {self.summary_file}")
