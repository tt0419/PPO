"""
station_coverage_calculator.py
署所ごとのカバー範囲（6分/13分圏内グリッド集合）を事前計算するモジュール

【目的】
- 各署所から到達可能なグリッド集合を事前計算
- ランダムサンプリングを廃止し、決定論的なカバレッジ計算を実現
- StateEncoderやRewardDesignerで使用

【使用方法】
    # 事前計算の実行
    python station_coverage_calculator.py
    
    # モジュールとして使用
    from station_coverage_calculator import StationCoverageCalculator
    calc = StationCoverageCalculator.load("station_coverage.npz")
    loss_6, loss_13 = calc.calculate_coverage_loss(station_h3, available_stations)
"""

import numpy as np
import pandas as pd
import json
import h3
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
import time


class StationCoverageCalculator:
    """
    署所カバー範囲の事前計算と、カバレッジ損失計算を行うクラス
    
    【データ構造】
    - station_h3_to_id: {署所H3 → 署所ID} のマッピング
    - coverage_6min: {署所ID → グリッドIDのset} 6分圏内カバー範囲
    - coverage_13min: {署所ID → グリッドIDのset} 13分圏内カバー範囲
    - all_grid_ids: 全グリッドIDの集合
    """
    
    # 時間閾値（秒）
    TIME_THRESHOLD_6MIN = 360
    TIME_THRESHOLD_13MIN = 780
    
    def __init__(self):
        """初期化"""
        self.station_h3_to_id: Dict[str, int] = {}
        self.station_id_to_h3: Dict[int, str] = {}
        self.coverage_6min: Dict[int, Set[int]] = {}
        self.coverage_13min: Dict[int, Set[int]] = {}
        self.all_grid_ids: Set[int] = set()
        self.grid_mapping: Dict[str, int] = {}  # H3 → grid_id
        self.is_computed = False
    
    def compute_coverage(self,
                         ambulance_data: pd.DataFrame,
                         travel_time_matrix: np.ndarray,
                         grid_mapping: Dict[str, int],
                         verbose: bool = True) -> None:
        """
        署所カバー範囲を計算
        
        Args:
            ambulance_data: 救急車データ（latitude, longitude列を含む）
            travel_time_matrix: 移動時間行列 [N_grids × N_grids]
            grid_mapping: {H3インデックス → グリッドID}
            verbose: 詳細ログ出力
        """
        if verbose:
            print("=" * 60)
            print("署所カバー範囲の事前計算を開始")
            print("=" * 60)
        
        start_time = time.time()
        
        self.grid_mapping = grid_mapping
        self.all_grid_ids = set(range(travel_time_matrix.shape[0]))
        
        # 1. 署所の一意なH3インデックスを収集
        station_h3_set = set()
        ambulance_to_station = {}  # 救急車ID → 署所H3
        
        for idx, row in ambulance_data.iterrows():
            try:
                lat = float(row['latitude'])
                lng = float(row['longitude'])
                station_h3 = h3.latlng_to_cell(lat, lng, 9)
                station_h3_set.add(station_h3)
                ambulance_to_station[idx] = station_h3
            except Exception as e:
                if verbose:
                    print(f"警告: 救急車{idx}のH3計算エラー: {e}")
        
        # 署所H3 → 署所ID のマッピングを作成
        for station_id, station_h3 in enumerate(sorted(station_h3_set)):
            self.station_h3_to_id[station_h3] = station_id
            self.station_id_to_h3[station_id] = station_h3
        
        if verbose:
            print(f"  救急車数: {len(ambulance_data)}台")
            print(f"  一意な署所数: {len(self.station_h3_to_id)}署")
            print(f"  グリッド数: {len(self.all_grid_ids)}")
        
        # 2. 各署所のカバー範囲を計算
        if verbose:
            print("\n各署所のカバー範囲を計算中...")
        
        for station_h3, station_id in self.station_h3_to_id.items():
            # 署所のグリッドインデックスを取得
            if station_h3 not in grid_mapping:
                if verbose:
                    print(f"  警告: 署所{station_h3}がgrid_mappingに存在しません")
                self.coverage_6min[station_id] = set()
                self.coverage_13min[station_id] = set()
                continue
            
            station_grid_idx = grid_mapping[station_h3]
            
            # この署所から各グリッドへの移動時間を取得
            travel_times = travel_time_matrix[station_grid_idx, :]
            
            # 6分圏内・13分圏内のグリッドIDを取得
            covered_6min = set(np.where(travel_times <= self.TIME_THRESHOLD_6MIN)[0])
            covered_13min = set(np.where(travel_times <= self.TIME_THRESHOLD_13MIN)[0])
            
            self.coverage_6min[station_id] = covered_6min
            self.coverage_13min[station_id] = covered_13min
            
            if verbose and station_id < 5:
                print(f"  署所{station_id} ({station_h3}): "
                      f"6分圏={len(covered_6min)}グリッド, "
                      f"13分圏={len(covered_13min)}グリッド")
        
        self.is_computed = True
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"\n事前計算完了 ({elapsed:.2f}秒)")
            print("=" * 60)
    
    def get_station_id(self, station_h3: str) -> Optional[int]:
        """署所H3から署所IDを取得"""
        return self.station_h3_to_id.get(station_h3)
    
    def get_station_h3(self, station_id: int) -> Optional[str]:
        """署所IDから署所H3を取得"""
        return self.station_id_to_h3.get(station_id)
    
    def calculate_system_coverage(self, 
                                  available_station_ids: Set[int]
                                  ) -> Tuple[Set[int], Set[int]]:
        """
        現在出場可能な署所集合から、システム全体のカバレッジを計算
        
        Args:
            available_station_ids: 利用可能な署所IDの集合
            
        Returns:
            (6分圏内カバー集合, 13分圏内カバー集合)
        """
        coverage_6min = set()
        coverage_13min = set()
        
        for station_id in available_station_ids:
            if station_id in self.coverage_6min:
                coverage_6min |= self.coverage_6min[station_id]
            if station_id in self.coverage_13min:
                coverage_13min |= self.coverage_13min[station_id]
        
        return coverage_6min, coverage_13min
    
    def calculate_coverage_loss(self,
                                departing_station_h3: str,
                                available_station_h3s: Set[str],
                                ambulances_per_station: Optional[Dict[str, int]] = None
                                ) -> Tuple[float, float]:
        """
        指定した署所から出場した場合のカバレッジ損失を計算
        
        Args:
            departing_station_h3: 出場する署所のH3
            available_station_h3s: 現在利用可能な署所H3の集合
            ambulances_per_station: 各署所の利用可能台数（Noneの場合は全て1台と仮定）
            
        Returns:
            (L6, L13): 6分/13分カバレッジ損失（0-1の正規化値）
        """
        if not self.is_computed:
            raise RuntimeError("事前計算が実行されていません。compute_coverage()を先に呼び出してください。")
        
        # 署所H3 → 署所IDに変換
        available_station_ids = set()
        for h3_idx in available_station_h3s:
            station_id = self.station_h3_to_id.get(h3_idx)
            if station_id is not None:
                available_station_ids.add(station_id)
        
        departing_station_id = self.station_h3_to_id.get(departing_station_h3)
        
        if departing_station_id is None:
            # 出場署所が不明な場合はデフォルト値
            return 0.5, 0.5
        
        # 現在のカバレッジを計算
        current_6min, current_13min = self.calculate_system_coverage(available_station_ids)
        
        # 出場後のカバレッジを計算
        # 署所に複数台ある場合は、1台出場しても署所のカバレッジは維持される
        remaining_station_ids = available_station_ids.copy()
        
        if ambulances_per_station is not None:
            # 台数情報がある場合
            station_h3 = departing_station_h3
            count = ambulances_per_station.get(station_h3, 1)
            if count <= 1:
                # この署所から最後の1台が出場する → カバレッジから除外
                remaining_station_ids.discard(departing_station_id)
            # count > 1 の場合は、まだ他の隊がいるので除外しない
        else:
            # 台数情報がない場合は、常に除外（保守的な計算）
            remaining_station_ids.discard(departing_station_id)
        
        remaining_6min, remaining_13min = self.calculate_system_coverage(remaining_station_ids)
        
        # カバレッジ損失を計算
        total_grids = len(self.all_grid_ids)
        if total_grids == 0:
            return 0.0, 0.0
        
        loss_6min = (len(current_6min) - len(remaining_6min)) / total_grids
        loss_13min = (len(current_13min) - len(remaining_13min)) / total_grids
        
        # 0-1にクリップ
        loss_6min = max(0.0, min(1.0, loss_6min))
        loss_13min = max(0.0, min(1.0, loss_13min))
        
        return loss_6min, loss_13min
    
    def calculate_coverage_loss_by_station_id(self,
                                              departing_station_id: int,
                                              available_station_ids: Set[int],
                                              ambulances_per_station_id: Optional[Dict[int, int]] = None
                                              ) -> Tuple[float, float]:
        """
        署所IDを使ってカバレッジ損失を計算（高速版）
        
        Args:
            departing_station_id: 出場する署所のID
            available_station_ids: 現在利用可能な署所IDの集合
            ambulances_per_station_id: 各署所IDの利用可能台数
            
        Returns:
            (L6, L13): 6分/13分カバレッジ損失
        """
        if not self.is_computed:
            raise RuntimeError("事前計算が実行されていません。")
        
        # 現在のカバレッジを計算
        current_6min, current_13min = self.calculate_system_coverage(available_station_ids)
        
        # 出場後のカバレッジを計算
        remaining_station_ids = available_station_ids.copy()
        
        if ambulances_per_station_id is not None:
            count = ambulances_per_station_id.get(departing_station_id, 1)
            if count <= 1:
                remaining_station_ids.discard(departing_station_id)
        else:
            remaining_station_ids.discard(departing_station_id)
        
        remaining_6min, remaining_13min = self.calculate_system_coverage(remaining_station_ids)
        
        # カバレッジ損失を計算
        total_grids = len(self.all_grid_ids)
        if total_grids == 0:
            return 0.0, 0.0
        
        loss_6min = (len(current_6min) - len(remaining_6min)) / total_grids
        loss_13min = (len(current_13min) - len(remaining_13min)) / total_grids
        
        return max(0.0, min(1.0, loss_6min)), max(0.0, min(1.0, loss_13min))
    
    def save(self, filepath: str) -> None:
        """
        計算結果をファイルに保存
        
        Args:
            filepath: 保存先パス（.npz形式推奨）
        """
        if not self.is_computed:
            raise RuntimeError("事前計算が実行されていません。")
        
        # SetをリストやNumPy配列に変換して保存
        # NumPy int64 をPythonネイティブ int に変換
        coverage_6min_dict = {
            str(k): [int(x) for x in v] 
            for k, v in self.coverage_6min.items()
        }
        coverage_13min_dict = {
            str(k): [int(x) for x in v] 
            for k, v in self.coverage_13min.items()
        }
        
        # station_h3_to_id の値もネイティブ int に変換
        station_h3_to_id_native = {
            str(k): int(v) for k, v in self.station_h3_to_id.items()
        }
        station_id_to_h3_native = {
            int(k): str(v) for k, v in self.station_id_to_h3.items()
        }
        
        # grid_mappingの値もネイティブ int に変換
        grid_mapping_native = {
            str(k): int(v) for k, v in self.grid_mapping.items()
        }
        
        data = {
            'station_h3_to_id': station_h3_to_id_native,
            'station_id_to_h3': station_id_to_h3_native,
            'coverage_6min': coverage_6min_dict,
            'coverage_13min': coverage_13min_dict,
            'all_grid_ids': [int(x) for x in self.all_grid_ids],
            'grid_mapping': grid_mapping_native
        }
        
        # JSONで保存（互換性のため）
        json_path = filepath.replace('.npz', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"保存完了: {json_path}")
    
    @classmethod
    def load(cls, filepath: str) -> 'StationCoverageCalculator':
        """
        ファイルから読み込み
        
        Args:
            filepath: 読み込み元パス
            
        Returns:
            StationCoverageCalculator インスタンス
        """
        json_path = filepath.replace('.npz', '.json')
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        calc = cls()
        calc.station_h3_to_id = data['station_h3_to_id']
        calc.station_id_to_h3 = {int(k): v for k, v in data['station_id_to_h3'].items()}
        calc.coverage_6min = {int(k): set(v) for k, v in data['coverage_6min'].items()}
        calc.coverage_13min = {int(k): set(v) for k, v in data['coverage_13min'].items()}
        calc.all_grid_ids = set(data['all_grid_ids'])
        calc.grid_mapping = data['grid_mapping']
        calc.is_computed = True
        
        print(f"読み込み完了: {json_path}")
        print(f"  署所数: {len(calc.station_h3_to_id)}")
        print(f"  グリッド数: {len(calc.all_grid_ids)}")
        
        return calc
    
    def get_statistics(self) -> Dict:
        """
        統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        if not self.is_computed:
            return {}
        
        coverage_6min_sizes = [len(v) for v in self.coverage_6min.values()]
        coverage_13min_sizes = [len(v) for v in self.coverage_13min.values()]
        
        return {
            'num_stations': len(self.station_h3_to_id),
            'num_grids': len(self.all_grid_ids),
            'coverage_6min': {
                'mean': np.mean(coverage_6min_sizes) if coverage_6min_sizes else 0,
                'min': min(coverage_6min_sizes) if coverage_6min_sizes else 0,
                'max': max(coverage_6min_sizes) if coverage_6min_sizes else 0,
            },
            'coverage_13min': {
                'mean': np.mean(coverage_13min_sizes) if coverage_13min_sizes else 0,
                'min': min(coverage_13min_sizes) if coverage_13min_sizes else 0,
                'max': max(coverage_13min_sizes) if coverage_13min_sizes else 0,
            }
        }


def main():
    """
    事前計算のスタンドアロン実行
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='署所カバー範囲の事前計算')
    parser.add_argument('--data-dir', type=str, default='data/tokyo',
                        help='データディレクトリのパス')
    parser.add_argument('--output', type=str, default='station_coverage.json',
                        help='出力ファイルパス')
    args = parser.parse_args()
    
    base_dir = Path(args.data_dir)
    
    print("=" * 60)
    print("署所カバー範囲の事前計算")
    print("=" * 60)
    
    # 1. 救急車データの読み込み
    print("\n1. 救急車データ読み込み...")
    firestation_path = base_dir / "import/amb_place_master.csv"
    ambulance_data = pd.read_csv(firestation_path, encoding='utf-8')
    ambulance_data = ambulance_data[ambulance_data['special_flag'] == 1]
    
    # フィルタリング（ValidationSimulatorと同じ）
    if 'team_name' in ambulance_data.columns:
        ambulance_data = ambulance_data[ambulance_data['team_name'] != '救急隊なし'].copy()
        ambulance_data = ambulance_data[~ambulance_data['team_name'].str.contains('デイタイム', na=False)].copy()
    
    print(f"  救急車数: {len(ambulance_data)}台")
    
    # 2. グリッドマッピングの読み込み
    print("\n2. グリッドマッピング読み込み...")
    grid_mapping_path = base_dir / "processed/grid_mapping_res9.json"
    with open(grid_mapping_path, 'r', encoding='utf-8') as f:
        grid_mapping = json.load(f)
    print(f"  グリッド数: {len(grid_mapping)}")
    
    # 3. 移動時間行列の読み込み
    print("\n3. 移動時間行列読み込み...")
    calibration_dir = base_dir / "calibration2"
    travel_time_path = calibration_dir / "linear_calibrated_response.npy"
    travel_time_matrix = np.load(travel_time_path)
    print(f"  行列サイズ: {travel_time_matrix.shape}")
    
    # 4. カバー範囲の計算
    print("\n4. カバー範囲計算...")
    calculator = StationCoverageCalculator()
    calculator.compute_coverage(
        ambulance_data=ambulance_data,
        travel_time_matrix=travel_time_matrix,
        grid_mapping=grid_mapping,
        verbose=True
    )
    
    # 5. 統計情報の表示
    print("\n5. 統計情報:")
    stats = calculator.get_statistics()
    print(f"  署所数: {stats['num_stations']}")
    print(f"  グリッド数: {stats['num_grids']}")
    print(f"  6分圏内グリッド数: 平均{stats['coverage_6min']['mean']:.1f}, "
          f"最小{stats['coverage_6min']['min']}, 最大{stats['coverage_6min']['max']}")
    print(f"  13分圏内グリッド数: 平均{stats['coverage_13min']['mean']:.1f}, "
          f"最小{stats['coverage_13min']['min']}, 最大{stats['coverage_13min']['max']}")
    
    # 6. 保存
    print(f"\n6. 保存: {args.output}")
    calculator.save(args.output)
    
    # 7. テスト：カバレッジ損失計算
    print("\n7. テスト: カバレッジ損失計算")
    
    # 全署所が利用可能な状態でのテスト
    all_station_h3s = set(calculator.station_h3_to_id.keys())
    test_station_h3 = list(all_station_h3s)[0]
    
    loss_6, loss_13 = calculator.calculate_coverage_loss(
        departing_station_h3=test_station_h3,
        available_station_h3s=all_station_h3s
    )
    
    print(f"  テスト署所: {test_station_h3}")
    print(f"  カバレッジ損失: L6={loss_6:.4f}, L13={loss_13:.4f}")
    
    print("\n" + "=" * 60)
    print("完了")
    print("=" * 60)


if __name__ == '__main__':
    main()
