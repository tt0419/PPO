"""
state_encoder.py
複雑な環境状態を固定長ベクトルに変換
移動時間行列を統合し、実際の道路網での移動時間を特徴量として使用
"""

import numpy as np
import h3
from typing import Dict, List, Optional
import torch
import sys
import os

# 統一された傷病度定数をインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from constants import SEVERITY_INDICES

class StateEncoder:
    """
    EMS環境の状態をニューラルネットワーク用のベクトルに変換
    実際の移動時間行列を使用して空間的関係を正確に表現
    """
    
    def __init__(self, config: Dict, max_ambulances: int = 192,
                 travel_time_matrix: Optional[np.ndarray] = None,
                 grid_mapping: Optional[Dict] = None):
        """
        Args:
            config: 設定辞書
            max_ambulances: 最大救急車数
            travel_time_matrix: responseフェーズの移動時間行列
            grid_mapping: H3インデックスから行列インデックスへのマッピング
        """
        self.config = config
        
        # 移動時間行列とグリッドマッピング
        self.travel_time_matrix = travel_time_matrix
        self.grid_mapping = grid_mapping
        
        # 特徴量の次元設定（動的に設定可能）
        self.max_ambulances = max_ambulances
        # 救急車特徴量を4→5次元に拡張（移動時間を追加）
        self.ambulance_features = 5  # 位置x, 位置y, 状態, 出動回数, 事案現場までの移動時間
        self.incident_features = 10  # 位置、傷病度など
        self.temporal_features = 8  # 時間関連
        # ★★★【修正箇所①】★★★
        # 空間特徴量の次元を1つ追加（カバレッジ率）
        self.spatial_features = 21  # 空間統計（改良版）+ カバレッジ率
        
        # 傷病度のone-hotエンコーディング用
        self.severity_indices = SEVERITY_INDICES
        
        # ★★★【修正箇所】★★★
        # コンフィグからカバレッジの時間閾値を読み込む
        coverage_config = config.get('coverage_params', {})
        self.coverage_time_threshold = coverage_config.get('time_threshold_seconds', 600)
        
    def encode_state(self, state_dict: Dict, grid_mapping: Dict = None) -> np.ndarray:
        """
        状態辞書を固定長ベクトルに変換
        
        Args:
            state_dict: 環境の状態情報
            grid_mapping: H3インデックスとグリッドIDのマッピング（後方互換性のため残す）
            
        Returns:
            状態ベクトル
        """
        # grid_mappingが引数で渡された場合はそれを使用（後方互換性）
        if grid_mapping is None:
            grid_mapping = self.grid_mapping
            
        features = []
        
        # 1. 救急車の特徴量（移動時間を含む拡張版）
        ambulance_features = self._encode_ambulances_with_travel_time(
            state_dict['ambulances'], 
            state_dict.get('pending_call'),
            grid_mapping
        )
        features.append(ambulance_features)
        
        # 2. 事案の特徴量
        incident_features = self._encode_incident(
            state_dict.get('pending_call'), grid_mapping
        )
        features.append(incident_features)
        
        # 3. 時間的特徴量
        temporal_features = self._encode_temporal(
            state_dict.get('episode_step', 0),
            state_dict.get('time_of_day', 12)
        )
        features.append(temporal_features)
        
        # ★★★【修正箇所②】★★★
        # 4. 空間的特徴量（カバレッジ率を追加）
        spatial_features = self._encode_spatial_with_coverage(
            state_dict['ambulances'],
            state_dict.get('pending_call'),
            grid_mapping
        )
        features.append(spatial_features)
        
        # 全特徴量を結合
        state_vector = np.concatenate(features)
        
        # NaN値のチェックと修正
        if np.any(np.isnan(state_vector)):
            print(f"警告: StateEncoderでNaN値を検出しました")
            state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1.0, neginf=0.0)
        
        return state_vector.astype(np.float32)
    
    def _encode_ambulances_with_travel_time(self, ambulances: Dict, 
                                           incident: Optional[Dict],
                                           grid_mapping: Dict) -> np.ndarray:
        """救急車情報をエンコード（移動時間を含む）"""
        # 動的に設定された台数を使用
        features = np.zeros(self.max_ambulances * self.ambulance_features)
        
        # 事案がある場合、その位置のグリッドインデックスを取得
        incident_grid_idx = None
        if incident is not None and self.travel_time_matrix is not None and grid_mapping:
            try:
                incident_h3 = incident.get('h3_index')
                if incident_h3 and incident_h3 in grid_mapping:
                    incident_grid_idx = grid_mapping[incident_h3]
            except Exception as e:
                print(f"警告: 事案位置のグリッドインデックス取得失敗: {e}")
        
        for amb_id, amb_state in ambulances.items():
            if amb_id >= self.max_ambulances:
                break
            
            idx = amb_id * self.ambulance_features
            
            # H3インデックスを座標に変換
            try:
                lat, lng = h3.cell_to_latlng(amb_state['current_h3'])
            except:
                lat, lng = 35.6762, 139.6503  # デフォルト（東京）
            
            # 基本特徴量の設定（安全な正規化）
            features[idx] = (lat + 90.0) / 180.0  # 緯度を[0, 1]に正規化
            features[idx + 1] = (lng + 180.0) / 360.0  # 経度を[0, 1]に正規化
            features[idx + 2] = 1.0 if amb_state['status'] == 'available' else 0.0
            features[idx + 3] = min(amb_state.get('calls_today', 0) / 20.0, 1.0)  # 出動回数を正規化
            
            # 新規追加：事案現場までの実際の移動時間
            travel_time_minutes = 0.0
            if incident_grid_idx is not None and self.travel_time_matrix is not None:
                try:
                    amb_h3 = amb_state.get('current_h3')
                    if amb_h3 and amb_h3 in grid_mapping:
                        amb_grid_idx = grid_mapping[amb_h3]
                        # 移動時間行列から実際の移動時間を取得（秒）
                        travel_time_seconds = self.travel_time_matrix[amb_grid_idx, incident_grid_idx]
                        # 分に変換して正規化（0-30分を0-1にマッピング）
                        travel_time_minutes = min(travel_time_seconds / 60.0 / 30.0, 1.0)
                except Exception as e:
                    # エラー時はデフォルト値を使用
                    travel_time_minutes = 0.5
            
            features[idx + 4] = travel_time_minutes
        
        return features
    
    def _encode_incident(self, incident: Optional[Dict], grid_mapping: Dict) -> np.ndarray:
        """事案情報をエンコード"""
        features = np.zeros(self.incident_features)
        
        if incident is None:
            return features
        
        # 位置情報
        try:
            lat, lng = h3.cell_to_latlng(incident['h3_index'])
            features[0] = (lat + 90.0) / 180.0
            features[1] = (lng + 180.0) / 360.0
        except:
            features[0] = 0.5
            features[1] = 0.5
        
        # 傷病度（one-hot encoding）
        severity = incident.get('severity', '軽症')
        if severity in self.severity_indices:
            severity_idx = self.severity_indices[severity]
            if 2 + severity_idx < len(features):
                features[2 + severity_idx] = 1.0
        
        # 待機時間（正規化）
        wait_time = incident.get('wait_time', 0)
        features[8] = min(wait_time / 600.0, 1.0)  # 10分を1.0とする
        
        # 優先度スコア
        priority = incident.get('priority', 0.5)
        features[9] = priority
        
        return features
    
    def _encode_temporal(self, episode_step: int, time_of_day: float) -> np.ndarray:
        """時間的特徴量をエンコード"""
        features = np.zeros(self.temporal_features)
        
        # エピソード進行度
        max_steps = self.config.get('data', {}).get('episode_duration_hours', 24) * 60
        features[0] = min(episode_step / max_steps, 1.0)
        
        # 時刻（周期的エンコーディング）
        hour = time_of_day % 24
        features[1] = np.sin(2 * np.pi * hour / 24)
        features[2] = np.cos(2 * np.pi * hour / 24)
        
        # 時間帯カテゴリ（朝、昼、夕、夜）
        if 6 <= hour < 10:
            features[3] = 1.0  # 朝
        elif 10 <= hour < 17:
            features[4] = 1.0  # 昼
        elif 17 <= hour < 21:
            features[5] = 1.0  # 夕
        else:
            features[6] = 1.0  # 夜
        
        # 曜日情報（仮定：平日）
        features[7] = 1.0  # 平日フラグ
        
        return features
    
    def _encode_spatial_with_travel_time(self, ambulances: Dict, 
                                        incident: Optional[Dict],
                                        grid_mapping: Dict) -> np.ndarray:
        """
        空間的特徴量をエンコード（移動時間行列を使用した改良版）
        実際の道路網での移動時間統計を計算
        """
        features = np.zeros(20)  # 元の20次元の特徴量
        
        if incident is None or self.travel_time_matrix is None or grid_mapping is None:
            # 移動時間行列が利用できない場合は従来の方法にフォールバック
            return self._encode_spatial_fallback(ambulances, incident)
        
        # 事案位置のグリッドインデックスを取得
        try:
            incident_h3 = incident.get('h3_index')
            if not incident_h3 or incident_h3 not in grid_mapping:
                return self._encode_spatial_fallback(ambulances, incident)
            
            incident_grid_idx = grid_mapping[incident_h3]
        except Exception as e:
            print(f"警告: 空間特徴量計算でエラー: {e}")
            return self._encode_spatial_fallback(ambulances, incident)
        
        # 利用可能な救急車の移動時間を収集
        available_times = []
        all_times = []
        
        for amb_id, amb_state in ambulances.items():
            try:
                amb_h3 = amb_state.get('current_h3')
                if amb_h3 and amb_h3 in grid_mapping:
                    amb_grid_idx = grid_mapping[amb_h3]
                    travel_time_seconds = self.travel_time_matrix[amb_grid_idx, incident_grid_idx]
                    travel_time_minutes = travel_time_seconds / 60.0
                    
                    all_times.append(travel_time_minutes)
                    
                    if amb_state['status'] == 'available':
                        available_times.append(travel_time_minutes)
            except:
                continue
        
        # 統計量を計算
        if available_times:
            # 利用可能な救急車の統計
            features[0] = min(available_times) / 30.0  # 最短時間（30分で正規化）
            features[1] = np.mean(available_times) / 30.0  # 平均時間
            features[2] = np.median(available_times) / 30.0  # 中央値
            features[3] = np.std(available_times) / 10.0 if len(available_times) > 1 else 0  # 標準偏差
            features[4] = len(available_times) / max(len(ambulances), 1)  # 利用可能率
            
            # 時間帯別カウント（5分、10分、15分、20分以内）
            features[5] = sum(1 for t in available_times if t <= 5) / max(len(available_times), 1)
            features[6] = sum(1 for t in available_times if t <= 10) / max(len(available_times), 1)
            features[7] = sum(1 for t in available_times if t <= 15) / max(len(available_times), 1)
            features[8] = sum(1 for t in available_times if t <= 20) / max(len(available_times), 1)
        
        if all_times:
            # 全救急車の統計
            features[9] = min(all_times) / 30.0  # 全体最短時間
            features[10] = np.mean(all_times) / 30.0  # 全体平均時間
            features[11] = np.median(all_times) / 30.0  # 全体中央値
            features[12] = max(all_times) / 60.0  # 最長時間（60分で正規化）
            
            # 分位数
            features[13] = np.percentile(all_times, 25) / 30.0  # 第1四分位
            features[14] = np.percentile(all_times, 75) / 30.0  # 第3四分位
            
        # 救急車の稼働状況
        total_ambulances = len(ambulances)
        if total_ambulances > 0:
            available_count = sum(1 for a in ambulances.values() if a['status'] == 'available')
            busy_count = total_ambulances - available_count
            
            features[15] = available_count / total_ambulances  # 利用可能率
            features[16] = busy_count / total_ambulances  # 稼働率
            features[17] = available_count / 20.0  # 絶対数（20台で正規化）
            features[18] = min(available_count / 5.0, 1.0)  # 5台以上で飽和
            features[19] = 1.0 if available_count > 0 else 0.0  # 利用可能フラグ
        
        return features
    
    def _encode_spatial_with_coverage(self, ambulances: Dict, 
                                    incident: Optional[Dict],
                                    grid_mapping: Dict) -> np.ndarray:
        """
        空間的特徴量をエンコード。最後にカバレッジ率を追加する。
        """
        # 既存の空間特徴量（20次元）を計算
        features = np.zeros(self.spatial_features)  # 21次元の配列を初期化
        
        # 既存の空間特徴量計算を呼び出して最初の20次元を埋める
        existing_features = self._encode_spatial_with_travel_time(
            ambulances, incident, grid_mapping
        )
        features[:20] = existing_features
        
        # --- 新しいカバレッジ特徴量の計算 ---
        # 1. 利用可能な救急隊のH3インデックスを取得
        available_amb_h3s = [
            amb_state['current_h3'] 
            for amb_state in ambulances.values() 
            if amb_state['status'] == 'available'
        ]

        # 2. カバレッジを計算
        coverage_ratio = 0.0
        if available_amb_h3s and self.travel_time_matrix is not None and grid_mapping:
            total_grids = len(grid_mapping)
            covered_grids = set()
            
            # 各利用可能隊から10分以内のグリッドを調べる
            for h3_index in available_amb_h3s:
                amb_grid_idx = grid_mapping.get(h3_index)
                if amb_grid_idx is None:
                    continue

                # 移動時間行列から、この救急隊からの移動時間リストを取得
                travel_times_from_amb = self.travel_time_matrix[amb_grid_idx, :]
                
                # ★★★【修正箇所】★★★
                # ハードコーディングされた600を、コンフィグから読み込んだ変数に置き換える
                covered_indices = np.where(travel_times_from_amb <= self.coverage_time_threshold)[0]
                
                # setに追加して重複を除外
                covered_grids.update(covered_indices)
            
            # 全グリッド数に対するカバーされたグリッド数の割合を計算
            if total_grids > 0:
                coverage_ratio = len(covered_grids) / total_grids
        
        # 計算したカバレッジ率を最後の特徴量として追加
        features[20] = coverage_ratio
        
        return features
    
    def _encode_spatial_fallback(self, ambulances: Dict, 
                                incident: Optional[Dict]) -> np.ndarray:
        """
        空間的特徴量をエンコード（フォールバック版）
        移動時間行列が利用できない場合の従来の実装
        """
        features = np.zeros(20)  # 元の20次元の特徴量
        
        if incident is None:
            return features
        
        # 事案位置
        try:
            incident_lat, incident_lng = h3.cell_to_latlng(incident['h3_index'])
        except:
            return features
        
        # 各救急車との距離を計算
        distances = []
        available_distances = []
        
        for amb_state in ambulances.values():
            try:
                lat, lng = h3.cell_to_latlng(amb_state['current_h3'])
                # Haversine距離（km）
                dist = self._haversine_distance(incident_lat, incident_lng, lat, lng)
                distances.append(dist)
                
                if amb_state['status'] == 'available':
                    available_distances.append(dist)
            except:
                continue
        
        # 統計量を計算
        if available_distances:
            features[0] = min(available_distances) / 10.0  # 最短距離
            features[1] = np.mean(available_distances) / 10.0
            features[2] = np.std(available_distances) / 5.0 if len(available_distances) > 1 else 0
            features[3] = len(available_distances) / 10.0  # 利用可能な救急車数
        
        if distances:
            features[4] = min(distances) / 10.0
            features[5] = np.mean(distances) / 10.0
        
        return features
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """2点間のHaversine距離を計算（km）"""
        R = 6371  # 地球の半径（km）
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    @property
    def state_dim(self) -> int:
        """状態ベクトルの次元数を返す"""
        return (self.max_ambulances * self.ambulance_features + 
                self.incident_features + 
                self.temporal_features + 
                self.spatial_features)


# ============================================================
# CompactStateEncoder - コンパクトな状態エンコーダー（46次元）
# ============================================================

class CompactStateEncoder:
    """
    コンパクトな状態エンコーダー（46次元）
    
    状態ベクトル構造:
    【候補隊情報】Top-10 × 4次元 = 40次元
      [i*4+0] 移動時間 (/ 30分)
      [i*4+1] 移動距離 (/ 10km)
      [i*4+2] カバレッジ損失 L6 (0-1)
      [i*4+3] カバレッジ損失 L13 (0-1)
    
    【グローバル状態】5次元
      [40] 利用可能率 (available / 192)
      [41] 出場中率 (dispatched / 192)
      [42] 6分圏内台数 (/ K)
      [43] 平均移動時間 (/ 30分)
      [44] システムカバレッジ C6
    
    【事案情報】1次元
      [45] 傷病度 (0=重症系, 1=軽症系)
    
    合計: 46次元
    """
    
    def __init__(self, 
                 config: Dict,
                 top_k: int = 10,
                 travel_time_matrix: Optional[np.ndarray] = None,
                 grid_mapping: Optional[Dict] = None):
        """
        Args:
            config: 設定辞書
            top_k: 考慮する上位救急車数（デフォルト10）
            travel_time_matrix: responseフェーズの移動時間行列
            grid_mapping: H3インデックス→行列インデックスのマッピング
        """
        self.config = config
        self.top_k = top_k
        self.travel_time_matrix = travel_time_matrix
        self.grid_mapping = grid_mapping
        
        # 特徴量の次元設定（仕様書に基づく）
        self.features_per_ambulance = 4   # 移動時間, 移動距離, L6, L13
        self.global_features = 5          # 利用可能率, 出場中率, 6分圏内, 平均移動, C6
        self.severity_features = 1        # 傷病度（binary）
        
        # 正規化パラメータ
        encoding_config = config.get('state_encoding', {}).get('normalization', {})
        self.max_travel_time_minutes = encoding_config.get('max_travel_time_minutes', 30)
        self.max_travel_distance_km = encoding_config.get('max_station_distance_km', 10)
        
        # カバレッジ計算の時間閾値（秒）
        self.time_threshold_6min = 360   # 6分
        self.time_threshold_13min = 780  # 13分
        
        # サンプリング設定（カバレッジ損失計算用）
        # ★★★ configから読み込むように変更 ★★★
        # coverage_aware_sorting または compact_coverage から読み込む
        coverage_config = config.get('state_encoding', {}).get('coverage_aware_sorting', {})
        if not coverage_config:
            coverage_config = config.get('state_encoding', {}).get('compact_coverage', {})
        self.coverage_sample_size = coverage_config.get('sample_size', 20)
        self.coverage_sample_radius = coverage_config.get('sample_radius', 2)
        
        # 傷病度判定用の定数
        self.severe_conditions = ['重症', '重篤', '死亡']
        self.mild_conditions = ['軽症', '中等症']
        
        # 総救急隊数（正規化用）
        self.total_ambulances = 192
        
        print(f"CompactStateEncoder初期化（46次元版）:")
        print(f"  Top-K: {top_k}")
        print(f"  状態次元: {self.state_dim}")
        print(f"  構成: 候補隊{top_k}×4={top_k*4}次元 + グローバル5次元 + 傷病度1次元")
        print(f"  カバレッジ計算: リング{self.coverage_sample_radius}, サンプル{self.coverage_sample_size}")
    
    @property
    def state_dim(self) -> int:
        """状態ベクトルの次元数"""
        return (self.top_k * self.features_per_ambulance) + self.global_features + self.severity_features
    
    def encode_state(self, state_dict: Dict, grid_mapping: Dict = None) -> np.ndarray:
        """
        状態辞書を46次元ベクトルに変換
        
        Args:
            state_dict: 環境の状態情報
                - 'ambulances': {amb_id: {'current_h3': str, 'status': str, 'station_h3': str, ...}}
                - 'pending_call': {'h3_index': str, 'severity': str, ...} or None
                - 'time_of_day': float (0-24)
            grid_mapping: H3→インデックスのマッピング（省略時はself.grid_mappingを使用）
        
        Returns:
            46次元のnumpy配列 (float32)
        """
        if grid_mapping is None:
            grid_mapping = self.grid_mapping
        
        features = np.zeros(self.state_dim, dtype=np.float32)
        
        incident = state_dict.get('pending_call')
        ambulances = state_dict.get('ambulances', {})
        
        # ========== 1. 候補隊情報（40次元）==========
        top_k_list = self._get_top_k_ambulances_with_coverage(ambulances, incident, grid_mapping)
        
        for i, amb_info in enumerate(top_k_list):
            base_idx = i * self.features_per_ambulance
            # 移動時間（正規化: /30分）
            features[base_idx + 0] = amb_info['travel_time_normalized']
            # 移動距離（正規化: /10km）
            features[base_idx + 1] = amb_info['travel_distance_normalized']
            # カバレッジ損失 L6（0-1）
            features[base_idx + 2] = amb_info['coverage_loss_6min']
            # カバレッジ損失 L13（0-1）
            features[base_idx + 3] = amb_info['coverage_loss_13min']
        
        # ========== 2. グローバル状態（5次元）==========
        global_base_idx = self.top_k * self.features_per_ambulance  # 40
        
        # 利用可能率
        available_count = sum(1 for a in ambulances.values() if a.get('status') == 'available')
        features[global_base_idx + 0] = available_count / self.total_ambulances
        
        # 出場中率
        dispatched_count = sum(1 for a in ambulances.values() if a.get('status') == 'dispatched')
        features[global_base_idx + 1] = dispatched_count / self.total_ambulances
        
        # 6分圏内台数（Top-K内）
        within_6min_count = sum(1 for a in top_k_list if a['travel_time_minutes'] <= 6)
        features[global_base_idx + 2] = within_6min_count / self.top_k
        
        # 平均移動時間（Top-K）
        valid_times = [a['travel_time_minutes'] for a in top_k_list if a.get('amb_id', -1) >= 0]
        if valid_times:
            avg_travel_time = np.mean(valid_times)
        else:
            avg_travel_time = self.max_travel_time_minutes
        features[global_base_idx + 3] = min(avg_travel_time / self.max_travel_time_minutes, 1.0)
        
        # システムカバレッジ C6
        features[global_base_idx + 4] = self._calculate_coverage_rate_6min(ambulances, grid_mapping)
        
        # ========== 3. 事案情報（1次元）==========
        severity_idx = global_base_idx + 5  # 45
        
        if incident:
            severity = incident.get('severity', '')
            # 重症系=0, 軽症系=1
            features[severity_idx] = 0.0 if severity in self.severe_conditions else 1.0
        else:
            features[severity_idx] = 0.5  # 事案なしの場合は中間値
        
        # NaNチェック
        if np.any(np.isnan(features)):
            print("警告: CompactStateEncoderでNaN値を検出")
            features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    def _get_top_k_ambulances_with_coverage(self, 
                                            ambulances: Dict, 
                                            incident: Optional[Dict],
                                            grid_mapping: Dict) -> List[Dict]:
        """
        移動時間順にTop-K救急車の情報を取得（カバレッジ損失L6, L13を計算）
        
        Returns:
            List[Dict]: 各要素は以下のキーを持つ
                - amb_id: int
                - travel_time_seconds: float
                - travel_time_minutes: float
                - travel_time_normalized: float (0-1)
                - travel_distance_km: float
                - travel_distance_normalized: float (0-1)
                - coverage_loss_6min: float (0-1)
                - coverage_loss_13min: float (0-1)
        """
        # 事案がない場合はダミーデータを返す
        if incident is None:
            return self._get_dummy_top_k()
        
        incident_h3 = incident.get('h3_index')
        if not incident_h3 or not grid_mapping or incident_h3 not in grid_mapping:
            return self._get_dummy_top_k()
        
        incident_grid_idx = grid_mapping[incident_h3]
        
        # 利用可能な救急車を収集
        candidates = []
        all_available = []
        
        for amb_id, amb_state in ambulances.items():
            if amb_state.get('status') != 'available':
                continue
            
            # 全利用可能救急車のリスト（カバレッジ損失計算用）
            all_available.append({
                'id': amb_id,
                'current_h3': amb_state.get('current_h3'),
                'station_h3': amb_state.get('station_h3')
            })
            
            # 移動時間を計算
            amb_h3 = amb_state.get('current_h3')
            if amb_h3 and grid_mapping and amb_h3 in grid_mapping and self.travel_time_matrix is not None:
                amb_grid_idx = grid_mapping[amb_h3]
                travel_time_seconds = self.travel_time_matrix[amb_grid_idx, incident_grid_idx]
            else:
                travel_time_seconds = 1800  # 30分（デフォルト）
            
            travel_time_minutes = travel_time_seconds / 60.0
            
            # 移動距離を計算（H3距離から推定）
            travel_distance_km = self._calculate_travel_distance(amb_state, incident)
            
            candidates.append({
                'amb_id': amb_id,
                'current_h3': amb_h3,
                'station_h3': amb_state.get('station_h3'),
                'travel_time_seconds': travel_time_seconds,
                'travel_time_minutes': travel_time_minutes,
                'travel_time_normalized': min(travel_time_minutes / self.max_travel_time_minutes, 1.0),
                'travel_distance_km': travel_distance_km,
                'travel_distance_normalized': min(travel_distance_km / self.max_travel_distance_km, 1.0),
                'coverage_loss_6min': 0.0,  # 後で計算
                'coverage_loss_13min': 0.0  # 後で計算
            })
        
        # 移動時間順でソート
        candidates.sort(key=lambda x: x['travel_time_seconds'])
        
        # Top-Kを取得
        top_k_candidates = candidates[:self.top_k]
        
        # 各候補にカバレッジ損失を計算
        for cand in top_k_candidates:
            L6, L13 = self.calculate_coverage_loss(
                {'id': cand['amb_id'], 'current_h3': cand['current_h3'], 'station_h3': cand['station_h3']},
                all_available,
                self.travel_time_matrix,
                grid_mapping
            )
            cand['coverage_loss_6min'] = L6
            cand['coverage_loss_13min'] = L13
        
        # Top-Kに満たない場合はダミーで埋める
        while len(top_k_candidates) < self.top_k:
            top_k_candidates.append(self._get_dummy_ambulance_info())
        
        return top_k_candidates
    
    def calculate_coverage_loss(self,
                               ambulance: Dict,
                               all_available: List[Dict],
                               travel_time_matrix: np.ndarray,
                               grid_mapping: Dict) -> tuple:
        """
        救急隊が出場した場合のカバレッジ損失を計算
        傷病度考慮運用（SeverityBasedStrategy）と同じロジック
        
        Args:
            ambulance: 評価対象の救急隊（dict with 'id', 'current_h3', 'station_h3'）
            all_available: 利用可能な全救急隊のリスト
            travel_time_matrix: 移動時間行列（numpy array）
            grid_mapping: H3インデックス→行列インデックスのマッピング（dict）
        
        Returns:
            L6: 6分カバレッジ損失 (0-1)
            L13: 13分カバレッジ損失 (0-1)
        """
        # この救急隊を除いた利用可能な救急隊リスト
        remaining = [amb for amb in all_available if amb['id'] != ambulance['id']]
        
        if not remaining:
            return 1.0, 1.0  # 他に救急隊がない場合は最大損失
        
        if travel_time_matrix is None or not grid_mapping:
            return 0.5, 0.5
        
        # サンプルポイントの取得（署所周辺、リング距離r=2以内）
        try:
            center_h3 = ambulance.get('station_h3') or ambulance.get('current_h3')
            if not center_h3:
                return 0.5, 0.5
            
            nearby_grids = h3.grid_disk(center_h3, self.coverage_sample_radius)
            sample_points = [g for g in nearby_grids if g in grid_mapping]
            
            # サンプルサイズを20に制限
            if len(sample_points) > self.coverage_sample_size:
                import random
                sample_points = random.sample(sample_points, self.coverage_sample_size)
        except Exception:
            return 0.5, 0.5
        
        if not sample_points:
            return 0.5, 0.5
        
        # 出場前後のカバレッジを計算
        coverage_6min_before = 0
        coverage_13min_before = 0
        coverage_6min_after = 0
        coverage_13min_after = 0
        
        for point_h3 in sample_points:
            point_idx = grid_mapping.get(point_h3)
            if point_idx is None:
                continue
            
            # 出場前：全救急隊での最小応答時間
            min_time_before = float('inf')
            for amb in all_available:
                amb_idx = grid_mapping.get(amb.get('current_h3'))
                if amb_idx is not None:
                    try:
                        travel_time = travel_time_matrix[amb_idx, point_idx]
                        min_time_before = min(min_time_before, travel_time)
                    except IndexError:
                        continue
            
            # 出場後：この救急隊を除いた最小応答時間
            min_time_after = float('inf')
            for amb in remaining:
                amb_idx = grid_mapping.get(amb.get('current_h3'))
                if amb_idx is not None:
                    try:
                        travel_time = travel_time_matrix[amb_idx, point_idx]
                        min_time_after = min(min_time_after, travel_time)
                    except IndexError:
                        continue
            
            # カバレッジのカウント
            if min_time_before <= self.time_threshold_6min:
                coverage_6min_before += 1
            if min_time_before <= self.time_threshold_13min:
                coverage_13min_before += 1
            if min_time_after <= self.time_threshold_6min:
                coverage_6min_after += 1
            if min_time_after <= self.time_threshold_13min:
                coverage_13min_after += 1
        
        # カバレッジ損失を計算
        total_points = len(sample_points)
        if total_points == 0:
            return 0.5, 0.5
        
        L6 = (coverage_6min_before - coverage_6min_after) / total_points
        L13 = (coverage_13min_before - coverage_13min_after) / total_points
        
        # 0-1の範囲にクリップ
        L6 = max(0.0, min(1.0, L6))
        L13 = max(0.0, min(1.0, L13))
        
        return L6, L13
    
    def _calculate_travel_distance(self, amb_state: Dict, incident: Optional[Dict]) -> float:
        """救急車から事案現場までの距離を計算（km）"""
        try:
            if incident is None:
                return 5.0  # デフォルト
            
            amb_h3 = amb_state.get('current_h3')
            incident_h3 = incident.get('h3_index')
            
            if amb_h3 and incident_h3:
                # H3インデックスから座標を取得
                amb_lat, amb_lng = h3.cell_to_latlng(amb_h3)
                incident_lat, incident_lng = h3.cell_to_latlng(incident_h3)
                
                # Haversine距離を計算
                return self._haversine_distance(amb_lat, amb_lng, incident_lat, incident_lng)
            return 5.0  # デフォルト
        except:
            return 5.0
    
    def _calculate_coverage_rate_6min(self, ambulances: Dict, grid_mapping: Dict) -> float:
        """現在のシステムカバレッジ率を計算（6分閾値）"""
        if not grid_mapping or self.travel_time_matrix is None:
            return 0.5
        
        try:
            # 利用可能な救急車のグリッドインデックスを取得
            available_indices = []
            for amb_state in ambulances.values():
                if amb_state.get('status') == 'available':
                    amb_h3 = amb_state.get('current_h3')
                    if amb_h3 and amb_h3 in grid_mapping:
                        available_indices.append(grid_mapping[amb_h3])
            
            if not available_indices:
                return 0.0
            
            # カバーされているグリッド数をカウント（6分閾値）
            total_grids = len(grid_mapping)
            covered_grids = set()
            
            for amb_idx in available_indices:
                # この救急車から6分以内のグリッドを取得
                travel_times = self.travel_time_matrix[amb_idx, :]
                covered_indices = np.where(travel_times <= self.time_threshold_6min)[0]
                covered_grids.update(covered_indices)
            
            return len(covered_grids) / total_grids if total_grids > 0 else 0.0
        
        except Exception as e:
            return 0.5
    
    def _get_dummy_top_k(self) -> List[Dict]:
        """ダミーのTop-Kリストを返す"""
        return [self._get_dummy_ambulance_info() for _ in range(self.top_k)]
    
    def _get_dummy_ambulance_info(self) -> Dict:
        """ダミーの救急車情報を返す"""
        return {
            'amb_id': -1,
            'travel_time_seconds': 1800,
            'travel_time_minutes': 30.0,
            'travel_time_normalized': 1.0,
            'travel_distance_km': 10.0,
            'travel_distance_normalized': 1.0,
            'coverage_loss_6min': 0.5,
            'coverage_loss_13min': 0.5
        }
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """2点間のHaversine距離を計算（km）"""
        R = 6371  # 地球の半径（km）
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_top_k_ambulance_ids(self, ambulances: Dict, incident: Optional[Dict]) -> List[int]:
        """
        Top-K救急車のIDリストを返す
        
        ems_environment.pyでactionを実際の救急車IDに変換するために使用
        
        Returns:
            List[int]: Top-K救急車のIDリスト（移動時間順）
        """
        top_k_list = self._get_top_k_ambulances_with_coverage(ambulances, incident, self.grid_mapping)
        return [amb['amb_id'] for amb in top_k_list if amb['amb_id'] >= 0]
    
    def get_selected_ambulance_coverage_loss(self, ambulances: Dict, incident: Optional[Dict], action: int) -> tuple:
        """
        選択された救急隊のカバレッジ損失を取得
        
        報酬計算で使用するために、ems_environment.pyから呼び出される
        
        Args:
            ambulances: 救急隊の状態辞書
            incident: 事案情報
            action: PPOが選択したアクション（0からTop_K-1）
        
        Returns:
            (L6, L13): カバレッジ損失のタプル
        """
        top_k_list = self._get_top_k_ambulances_with_coverage(ambulances, incident, self.grid_mapping)
        
        if 0 <= action < len(top_k_list):
            selected = top_k_list[action]
            return selected.get('coverage_loss_6min', 0.5), selected.get('coverage_loss_13min', 0.5)
        else:
            return 0.5, 0.5


# ============================================================
# ファクトリ関数
# ============================================================

def create_state_encoder(config: Dict, **kwargs):
    """
    設定に応じてStateEncoderを作成するファクトリ関数
    
    Args:
        config: 設定辞書
        **kwargs: travel_time_matrix, grid_mapping など
    
    Returns:
        StateEncoder または CompactStateEncoder
    """
    encoding_config = config.get('state_encoding', {})
    mode = encoding_config.get('mode', 'full')
    
    if mode == 'compact':
        top_k = encoding_config.get('top_k', 10)
        return CompactStateEncoder(config, top_k=top_k, **kwargs)
    else:
        # 既存のStateEncoderを使用
        max_ambulances = kwargs.pop('max_ambulances', 192)
        return StateEncoder(config, max_ambulances=max_ambulances, **kwargs)