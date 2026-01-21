# state_encoder.py 変更ガイド

## 概要

既存の`StateEncoder`クラス（999次元）を維持しつつ、新しい`CompactStateEncoder`クラス（37次元）を追加する。

---

## 既存コードの確認ポイント

以下のメソッドは既存のStateEncoderから流用・参考にできる:

1. `_calculate_travel_time` に相当する処理（travel_time_matrixを使用）
2. `grid_mapping` の使い方
3. `h3.cell_to_latlng` の使い方

---

## 追加するクラス: CompactStateEncoder

```python
class CompactStateEncoder:
    """
    コンパクトな状態エンコーダー（37次元）
    
    状態ベクトル構造:
    [0]     is_severe: 重症系フラグ（重症/重篤/死亡 = 1.0）
    [1]     is_mild: 軽症系フラグ（軽症/中等症 = 1.0）
    [2-4]   Top-1救急車: travel_time, coverage_loss, station_distance
    [5-7]   Top-2救急車: travel_time, coverage_loss, station_distance
    ...
    [29-31] Top-10救急車: travel_time, coverage_loss, station_distance
    [32]    available_count_normalized: 利用可能救急車数 / 192
    [33]    coverage_rate: 現在のカバレッジ率（0-1）
    [34]    time_of_day_normalized: 時刻 / 24
    [35]    within_6min_ratio: 6分以内到達可能な救急車の割合
    [36]    avg_travel_time_normalized: Top-10の平均移動時間 / 30分
    """
    
    def __init__(self, 
                 config: Dict,
                 top_k: int = 10,
                 travel_time_matrix: Optional[np.ndarray] = None,
                 grid_mapping: Optional[Dict] = None):
        """
        Args:
            config: 設定辞書
            top_k: 考慮する上位救急車数
            travel_time_matrix: responseフェーズの移動時間行列
            grid_mapping: H3インデックス→行列インデックスのマッピング
        """
        self.config = config
        self.top_k = top_k
        self.travel_time_matrix = travel_time_matrix
        self.grid_mapping = grid_mapping
        
        # 特徴量の次元設定
        self.severity_features = 2       # is_severe, is_mild
        self.features_per_ambulance = 3  # travel_time, coverage_loss, station_distance
        self.global_features = 5         # 5つのグローバル統計
        
        # 正規化パラメータ
        encoding_config = config.get('state_encoding', {}).get('normalization', {})
        self.max_travel_time_minutes = encoding_config.get('max_travel_time_minutes', 30)
        self.max_station_distance_km = encoding_config.get('max_station_distance_km', 10)
        
        # カバレッジ計算の時間閾値（秒）
        coverage_config = config.get('coverage_params', {})
        self.coverage_time_threshold = coverage_config.get('time_threshold_seconds', 600)
        
        print(f"CompactStateEncoder初期化: top_k={top_k}, state_dim={self.state_dim}")
    
    @property
    def state_dim(self) -> int:
        """状態ベクトルの次元数"""
        return self.severity_features + (self.top_k * self.features_per_ambulance) + self.global_features
    
    def encode_state(self, state_dict: Dict, grid_mapping: Dict = None) -> np.ndarray:
        """
        状態辞書を37次元ベクトルに変換
        
        Args:
            state_dict: 環境の状態情報
                - 'ambulances': {amb_id: {'current_h3': str, 'status': str, 'station_h3': str, ...}}
                - 'pending_call': {'h3_index': str, 'severity': str, ...} or None
                - 'time_of_day': float (0-24)
            grid_mapping: H3→インデックスのマッピング（省略時はself.grid_mappingを使用）
        
        Returns:
            37次元のnumpy配列 (float32)
        """
        if grid_mapping is None:
            grid_mapping = self.grid_mapping
        
        features = np.zeros(self.state_dim, dtype=np.float32)
        
        incident = state_dict.get('pending_call')
        ambulances = state_dict.get('ambulances', {})
        time_of_day = state_dict.get('time_of_day', 12.0)
        
        # ========== 1. 傷病度（2次元）==========
        if incident:
            severity = incident.get('severity', '')
            features[0] = 1.0 if severity in ['重症', '重篤', '死亡'] else 0.0  # is_severe
            features[1] = 1.0 if severity in ['軽症', '中等症'] else 0.0        # is_mild
        
        # ========== 2. Top-K救急車（30次元）==========
        top_k_list = self._get_top_k_ambulances(ambulances, incident, grid_mapping)
        
        for i, amb_info in enumerate(top_k_list):
            base_idx = self.severity_features + i * self.features_per_ambulance
            features[base_idx + 0] = amb_info['travel_time_normalized']
            features[base_idx + 1] = amb_info['coverage_loss']
            features[base_idx + 2] = amb_info['station_distance_normalized']
        
        # ========== 3. グローバル統計（5次元）==========
        global_idx = self.severity_features + self.top_k * self.features_per_ambulance
        
        # 利用可能救急車数
        available_count = sum(1 for a in ambulances.values() if a.get('status') == 'available')
        features[global_idx + 0] = available_count / 192.0
        
        # カバレッジ率
        features[global_idx + 1] = self._calculate_coverage_rate(ambulances, grid_mapping)
        
        # 時刻
        features[global_idx + 2] = time_of_day / 24.0
        
        # 6分以内到達可能な救急車の割合（Top-K内）
        within_6min_count = sum(1 for a in top_k_list if a['travel_time_minutes'] <= 6)
        features[global_idx + 3] = within_6min_count / self.top_k
        
        # 平均移動時間
        avg_travel_time = np.mean([a['travel_time_minutes'] for a in top_k_list])
        features[global_idx + 4] = min(avg_travel_time / self.max_travel_time_minutes, 1.0)
        
        # NaNチェック
        if np.any(np.isnan(features)):
            print("警告: CompactStateEncoderでNaN値を検出")
            features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    def _get_top_k_ambulances(self, 
                              ambulances: Dict, 
                              incident: Optional[Dict],
                              grid_mapping: Dict) -> List[Dict]:
        """
        移動時間順にTop-K救急車の情報を取得
        
        Returns:
            List[Dict]: 各要素は以下のキーを持つ
                - amb_id: int
                - travel_time_seconds: float
                - travel_time_minutes: float
                - travel_time_normalized: float (0-1)
                - coverage_loss: float (0-1)
                - station_distance_km: float
                - station_distance_normalized: float (0-1)
        """
        # 事案がない場合はダミーデータを返す
        if incident is None:
            return self._get_dummy_top_k()
        
        incident_h3 = incident.get('h3_index')
        if not incident_h3 or incident_h3 not in grid_mapping:
            return self._get_dummy_top_k()
        
        incident_grid_idx = grid_mapping[incident_h3]
        
        # 利用可能な救急車を収集
        candidates = []
        available_amb_ids = []
        
        for amb_id, amb_state in ambulances.items():
            if amb_state.get('status') != 'available':
                continue
            
            available_amb_ids.append(amb_id)
            
            # 移動時間を計算
            amb_h3 = amb_state.get('current_h3')
            if amb_h3 and amb_h3 in grid_mapping:
                amb_grid_idx = grid_mapping[amb_h3]
                travel_time_seconds = self.travel_time_matrix[amb_grid_idx, incident_grid_idx]
            else:
                travel_time_seconds = 1800  # 30分（デフォルト）
            
            travel_time_minutes = travel_time_seconds / 60.0
            
            # 署からの距離を計算
            station_distance_km = self._calculate_station_distance(amb_state)
            
            candidates.append({
                'amb_id': amb_id,
                'travel_time_seconds': travel_time_seconds,
                'travel_time_minutes': travel_time_minutes,
                'travel_time_normalized': min(travel_time_minutes / self.max_travel_time_minutes, 1.0),
                'coverage_loss': 0.0,  # 後で計算
                'station_distance_km': station_distance_km,
                'station_distance_normalized': min(station_distance_km / self.max_station_distance_km, 1.0)
            })
        
        # 移動時間順にソート
        candidates.sort(key=lambda x: x['travel_time_seconds'])
        
        # Top-Kを取得
        top_k_candidates = candidates[:self.top_k]
        
        # カバレッジ損失を計算（Top-Kのみ）
        for cand in top_k_candidates:
            cand['coverage_loss'] = self._calculate_coverage_loss_simple(
                cand['amb_id'], 
                ambulances, 
                available_amb_ids
            )
        
        # Top-Kに満たない場合はダミーで埋める
        while len(top_k_candidates) < self.top_k:
            top_k_candidates.append(self._get_dummy_ambulance_info())
        
        return top_k_candidates
    
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
            'coverage_loss': 0.5,
            'station_distance_km': 5.0,
            'station_distance_normalized': 0.5
        }
    
    def _calculate_station_distance(self, amb_state: Dict) -> float:
        """署からの距離を計算（km）"""
        try:
            current_h3 = amb_state.get('current_h3')
            station_h3 = amb_state.get('station_h3')
            
            if current_h3 and station_h3:
                # H3インデックスから座標を取得
                current_lat, current_lng = h3.cell_to_latlng(current_h3)
                station_lat, station_lng = h3.cell_to_latlng(station_h3)
                
                # Haversine距離を計算
                return self._haversine_distance(current_lat, current_lng, station_lat, station_lng)
            return 0.0
        except:
            return 0.0
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """2点間のHaversine距離を計算（km）"""
        R = 6371  # 地球の半径（km）
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _calculate_coverage_loss_simple(self, 
                                        amb_id: int, 
                                        ambulances: Dict,
                                        available_amb_ids: List[int]) -> float:
        """
        簡易版カバレッジ損失計算
        
        考え方: この救急車を出動させた場合、近隣の利用可能救急車数が減る
        近隣に他の救急車が多いほど、損失は小さい
        """
        try:
            amb_state = ambulances.get(amb_id)
            if not amb_state:
                return 0.5
            
            amb_h3 = amb_state.get('current_h3')
            if not amb_h3 or amb_h3 not in self.grid_mapping:
                return 0.5
            
            amb_grid_idx = self.grid_mapping[amb_h3]
            
            # 10分以内に到達可能な他の利用可能救急車をカウント
            nearby_count = 0
            for other_id in available_amb_ids:
                if other_id == amb_id:
                    continue
                
                other_state = ambulances.get(other_id)
                if not other_state:
                    continue
                
                other_h3 = other_state.get('current_h3')
                if not other_h3 or other_h3 not in self.grid_mapping:
                    continue
                
                other_grid_idx = self.grid_mapping[other_h3]
                
                # 2台の救急車間の移動時間を確認
                travel_time = self.travel_time_matrix[amb_grid_idx, other_grid_idx]
                if travel_time <= self.coverage_time_threshold:
                    nearby_count += 1
            
            # 近隣救急車が多いほど損失は小さい
            # nearby_count = 0 → loss = 1.0
            # nearby_count = 5 → loss ≈ 0.17
            return 1.0 / (nearby_count + 1)
        
        except Exception as e:
            return 0.5
    
    def _calculate_coverage_rate(self, ambulances: Dict, grid_mapping: Dict) -> float:
        """現在のカバレッジ率を計算"""
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
            
            # カバーされているグリッド数をカウント
            total_grids = len(grid_mapping)
            covered_grids = set()
            
            for amb_idx in available_indices:
                # この救急車から閾値時間以内のグリッドを取得
                travel_times = self.travel_time_matrix[amb_idx, :]
                covered_indices = np.where(travel_times <= self.coverage_time_threshold)[0]
                covered_grids.update(covered_indices)
            
            return len(covered_grids) / total_grids if total_grids > 0 else 0.0
        
        except Exception as e:
            return 0.5
    
    def get_top_k_ambulance_ids(self, ambulances: Dict, incident: Optional[Dict]) -> List[int]:
        """
        Top-K救急車のIDリストを返す
        
        ems_environment.pyでactionを実際の救急車IDに変換するために使用
        
        Returns:
            List[int]: Top-K救急車のIDリスト（移動時間順）
        """
        top_k_list = self._get_top_k_ambulances(ambulances, incident, self.grid_mapping)
        return [amb['amb_id'] for amb in top_k_list if amb['amb_id'] >= 0]


# ========== ファクトリ関数 ==========

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
```

---

## 必要なインポート

ファイルの先頭に以下を追加（既存のインポートに追加）:

```python
from typing import Dict, List, Optional
import h3
```

---

## 既存のStateEncoderとの共存

- 既存の`StateEncoder`クラスは**変更しない**
- 新しい`CompactStateEncoder`クラスを**追加**する
- `create_state_encoder`ファクトリ関数で切り替え

---

## テスト用コード

```python
# ファイル末尾に追加（開発時のみ使用）

if __name__ == "__main__":
    # テスト用の設定
    test_config = {
        'state_encoding': {
            'mode': 'compact',
            'top_k': 10
        }
    }
    
    # エンコーダーの初期化テスト
    encoder = CompactStateEncoder(test_config, top_k=10)
    print(f"State dim: {encoder.state_dim}")
    assert encoder.state_dim == 37, f"Expected 37, got {encoder.state_dim}"
    
    # ダミーデータでのエンコードテスト
    dummy_state = {
        'ambulances': {},
        'pending_call': {'h3_index': 'dummy', 'severity': '軽症'},
        'time_of_day': 12.0
    }
    
    state_vector = encoder.encode_state(dummy_state)
    print(f"State vector shape: {state_vector.shape}")
    assert state_vector.shape == (37,), f"Expected (37,), got {state_vector.shape}"
    
    print("All tests passed!")
```
