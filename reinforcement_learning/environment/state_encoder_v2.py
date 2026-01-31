"""
state_encoder_v2.py
新しいカバレッジ計算方式を使用するCompactStateEncoder

【変更点】
- ランダムサンプリングを廃止
- 事前計算したカバー範囲を使用（StationCoverageCalculator）
- 決定論的なカバレッジ損失計算

【互換性】
- 既存のCompactStateEncoderと同じインターフェース
- state_dimは同じ（46次元）
"""

import numpy as np
import h3
from typing import Dict, List, Optional, Set
from collections import defaultdict
import sys
import os

# 統一された傷病度定数をインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from constants import SEVERITY_INDICES
except ImportError:
    # フォールバック
    SEVERITY_INDICES = {
        '重症': 0, '重篤': 1, '死亡': 2,
        '中等症': 3, '軽症': 4
    }


class CompactStateEncoderV2:
    """
    コンパクトな状態エンコーダー v2（決定論的カバレッジ計算）
    
    状態ベクトル構造:
    【候補隊情報】Top-K × 4次元 = 40次元
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
                 grid_mapping: Optional[Dict] = None,
                 station_coverage_calculator = None):
        """
        Args:
            config: 設定辞書
            top_k: 考慮する上位救急車数（デフォルト10）
            travel_time_matrix: responseフェーズの移動時間行列
            grid_mapping: H3インデックス→行列インデックスのマッピング
            station_coverage_calculator: StationCoverageCalculatorインスタンス（オプション）
        """
        self.config = config
        self.top_k = top_k
        self.travel_time_matrix = travel_time_matrix
        self.grid_mapping = grid_mapping
        
        # 新しいカバレッジ計算モジュール
        self.station_coverage_calculator = station_coverage_calculator
        
        # 特徴量の次元設定
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
        
        # 傷病度判定用の定数
        self.severe_conditions = ['重症', '重篤', '死亡']
        self.mild_conditions = ['軽症', '中等症']
        
        # 総救急隊数（正規化用）
        self.total_ambulances = 192
        
        # カバレッジ計算モードのログ
        if self.station_coverage_calculator is not None:
            print(f"CompactStateEncoderV2初期化（決定論的カバレッジ計算）:")
            print(f"  Top-K: {top_k}")
            print(f"  状態次元: {self.state_dim}")
            print(f"  カバレッジ計算: 事前計算方式（ノイズなし）")
        else:
            print(f"CompactStateEncoderV2初期化（フォールバック: 旧方式）:")
            print(f"  Top-K: {top_k}")
            print(f"  状態次元: {self.state_dim}")
            print(f"  警告: StationCoverageCalculatorが設定されていません")
    
    @property
    def state_dim(self) -> int:
        """状態ベクトルの次元数"""
        return (self.top_k * self.features_per_ambulance) + self.global_features + self.severity_features
    
    def encode_state(self, state_dict: Dict, grid_mapping: Dict = None) -> np.ndarray:
        """
        状態辞書を46次元ベクトルに変換
        
        Args:
            state_dict: 環境の状態情報
            grid_mapping: H3→インデックスのマッピング
        
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
            features[base_idx + 0] = amb_info['travel_time_normalized']
            features[base_idx + 1] = amb_info['travel_distance_normalized']
            features[base_idx + 2] = amb_info['coverage_loss_6min']
            features[base_idx + 3] = amb_info['coverage_loss_13min']
        
        # ========== 2. グローバル状態（5次元）==========
        global_base_idx = self.top_k * self.features_per_ambulance
        
        # 利用可能数・出場中数のカウント
        available_count = 0
        dispatched_count = 0
        for amb_state in ambulances.values():
            status = amb_state.get('status', 'unknown')
            if status == 'available':
                available_count += 1
            elif status in ['dispatched', 'on_scene', 'transporting', 'at_hospital', 'returning']:
                dispatched_count += 1
        
        features[global_base_idx + 0] = available_count / self.total_ambulances
        features[global_base_idx + 1] = dispatched_count / self.total_ambulances
        
        # 6分圏内台数（Top-K内で6分以内の候補数）
        count_6min = sum(1 for amb in top_k_list 
                        if amb['travel_time_minutes'] <= 6.0 and amb['amb_id'] >= 0)
        features[global_base_idx + 2] = count_6min / self.top_k
        
        # 平均移動時間
        valid_times = [amb['travel_time_minutes'] for amb in top_k_list if amb['amb_id'] >= 0]
        if valid_times:
            features[global_base_idx + 3] = np.mean(valid_times) / self.max_travel_time_minutes
        
        # システムカバレッジ（6分圏内カバー率）
        if self.station_coverage_calculator is not None:
            available_station_ids = self._get_available_station_ids(ambulances)
            current_6min, _ = self.station_coverage_calculator.calculate_system_coverage(available_station_ids)
            total_grids = len(self.station_coverage_calculator.all_grid_ids)
            features[global_base_idx + 4] = len(current_6min) / total_grids if total_grids > 0 else 0.5
        else:
            features[global_base_idx + 4] = 0.5  # フォールバック
        
        # ========== 3. 事案情報（1次元）==========
        severity_idx = global_base_idx + self.global_features
        if incident is not None:
            severity = incident.get('severity', '軽症')
            features[severity_idx] = 0.0 if severity in self.severe_conditions else 1.0
        
        # NaN値のチェック
        if np.any(np.isnan(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        return features
    
    def _get_top_k_ambulances_with_coverage(self, 
                                            ambulances: Dict, 
                                            incident: Optional[Dict],
                                            grid_mapping: Dict) -> List[Dict]:
        """
        移動時間順にTop-K救急車の情報を取得（カバレッジ損失を計算）
        
        Returns:
            List[Dict]: 各要素は以下のキーを持つ
                - amb_id: int
                - station_h3: str
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
        
        # 署所ごとの利用可能台数をカウント
        ambulances_per_station: Dict[str, int] = defaultdict(int)
        available_station_h3s: Set[str] = set()
        
        for amb_id, amb_state in ambulances.items():
            if amb_state.get('status') != 'available':
                continue
            
            station_h3 = amb_state.get('station_h3')
            if station_h3:
                ambulances_per_station[station_h3] += 1
                available_station_h3s.add(station_h3)
            
            # 移動時間を計算
            amb_h3 = amb_state.get('current_h3')
            if amb_h3 and grid_mapping and amb_h3 in grid_mapping and self.travel_time_matrix is not None:
                amb_grid_idx = grid_mapping[amb_h3]
                travel_time_seconds = self.travel_time_matrix[amb_grid_idx, incident_grid_idx]
            else:
                travel_time_seconds = 1800  # 30分（デフォルト）
            
            travel_time_minutes = travel_time_seconds / 60.0
            travel_distance_km = self._calculate_travel_distance(amb_state, incident)
            
            candidates.append({
                'amb_id': amb_id,
                'current_h3': amb_h3,
                'station_h3': station_h3,
                'travel_time_seconds': travel_time_seconds,
                'travel_time_minutes': travel_time_minutes,
                'travel_time_normalized': min(travel_time_minutes / self.max_travel_time_minutes, 1.0),
                'travel_distance_km': travel_distance_km,
                'travel_distance_normalized': min(travel_distance_km / self.max_travel_distance_km, 1.0),
                'coverage_loss_6min': 0.0,
                'coverage_loss_13min': 0.0
            })
        
        # 移動時間順でソート
        candidates.sort(key=lambda x: x['travel_time_seconds'])
        
        # Top-Kを取得
        top_k_candidates = candidates[:self.top_k]
        
        # 各候補にカバレッジ損失を計算
        for cand in top_k_candidates:
            L6, L13 = self._calculate_coverage_loss_v2(
                cand['station_h3'],
                available_station_h3s,
                ambulances_per_station
            )
            cand['coverage_loss_6min'] = L6
            cand['coverage_loss_13min'] = L13
        
        # Top-Kに満たない場合はダミーで埋める
        while len(top_k_candidates) < self.top_k:
            top_k_candidates.append(self._get_dummy_ambulance_info())
        
        return top_k_candidates
    
    def _calculate_coverage_loss_v2(self,
                                    station_h3: str,
                                    available_station_h3s: Set[str],
                                    ambulances_per_station: Dict[str, int]) -> tuple:
        """
        新しいカバレッジ損失計算（決定論的）
        
        Args:
            station_h3: 出場する署所のH3
            available_station_h3s: 利用可能な署所H3の集合
            ambulances_per_station: 各署所の利用可能台数
        
        Returns:
            (L6, L13): カバレッジ損失
        """
        if self.station_coverage_calculator is None:
            # フォールバック: 旧方式（ただしランダムなしで固定値）
            return 0.1, 0.1
        
        if not station_h3:
            return 0.5, 0.5
        
        try:
            L6, L13 = self.station_coverage_calculator.calculate_coverage_loss(
                departing_station_h3=station_h3,
                available_station_h3s=available_station_h3s,
                ambulances_per_station=ambulances_per_station
            )
            return L6, L13
        except Exception as e:
            print(f"警告: カバレッジ損失計算エラー: {e}")
            return 0.5, 0.5
    
    def _get_available_station_ids(self, ambulances: Dict) -> Set[int]:
        """利用可能な署所IDの集合を取得"""
        if self.station_coverage_calculator is None:
            return set()
        
        station_ids = set()
        for amb_state in ambulances.values():
            if amb_state.get('status') == 'available':
                station_h3 = amb_state.get('station_h3')
                if station_h3:
                    station_id = self.station_coverage_calculator.get_station_id(station_h3)
                    if station_id is not None:
                        station_ids.add(station_id)
        return station_ids
    
    def _calculate_travel_distance(self, amb_state: Dict, incident: Optional[Dict]) -> float:
        """救急車から事案現場までの距離を計算（km）"""
        try:
            if incident is None:
                return 5.0
            
            amb_h3 = amb_state.get('current_h3')
            incident_h3 = incident.get('h3_index')
            
            if amb_h3 and incident_h3:
                amb_lat, amb_lng = h3.cell_to_latlng(amb_h3)
                incident_lat, incident_lng = h3.cell_to_latlng(incident_h3)
                return self._haversine_distance(amb_lat, amb_lng, incident_lat, incident_lng)
            return 5.0
        except:
            return 5.0
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """2点間のHaversine距離を計算（km）"""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def _get_dummy_top_k(self) -> List[Dict]:
        """ダミーのTop-Kリストを返す"""
        return [self._get_dummy_ambulance_info() for _ in range(self.top_k)]
    
    def _get_dummy_ambulance_info(self) -> Dict:
        """ダミーの救急車情報を返す"""
        return {
            'amb_id': -1,
            'station_h3': None,
            'travel_time_seconds': 1800,
            'travel_time_minutes': 30.0,
            'travel_time_normalized': 1.0,
            'travel_distance_km': 10.0,
            'travel_distance_normalized': 1.0,
            'coverage_loss_6min': 0.5,
            'coverage_loss_13min': 0.5
        }
    
    def get_top_k_ambulance_ids(self, ambulances: Dict, incident: Optional[Dict]) -> List[int]:
        """Top-K救急車のIDリストを返す"""
        top_k_list = self._get_top_k_ambulances_with_coverage(ambulances, incident, self.grid_mapping)
        return [amb['amb_id'] for amb in top_k_list if amb['amb_id'] >= 0]
    
    def get_selected_ambulance_coverage_loss(self, ambulances: Dict, incident: Optional[Dict], action: int) -> tuple:
        """選択された救急隊のカバレッジ損失を取得（報酬計算用）"""
        top_k_list = self._get_top_k_ambulances_with_coverage(ambulances, incident, self.grid_mapping)
        
        if 0 <= action < len(top_k_list):
            selected = top_k_list[action]
            return selected.get('coverage_loss_6min', 0.5), selected.get('coverage_loss_13min', 0.5)
        else:
            return 0.5, 0.5


# ============================================================
# ファクトリ関数
# ============================================================

def create_state_encoder_v2(config: Dict, 
                            station_coverage_calculator=None,
                            **kwargs):
    """
    設定に応じてStateEncoderV2を作成するファクトリ関数
    
    Args:
        config: 設定辞書
        station_coverage_calculator: StationCoverageCalculatorインスタンス
        **kwargs: travel_time_matrix, grid_mapping など
    
    Returns:
        CompactStateEncoderV2
    """
    encoding_config = config.get('state_encoding', {})
    top_k = encoding_config.get('top_k', 10)
    
    return CompactStateEncoderV2(
        config, 
        top_k=top_k, 
        station_coverage_calculator=station_coverage_calculator,
        **kwargs
    )
