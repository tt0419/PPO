"""
v13_integration_guide.py
新しいカバレッジ計算方式の統合ガイドとテストスクリプト

【このファイルの目的】
1. 新しいモジュールの動作確認
2. EMSEnvironmentへの統合方法の説明
3. 実際の統合コード例の提供

【統合手順】
1. station_coverage_calculator.py を reinforcement_learning/environment/ に配置
2. state_encoder_v2.py を reinforcement_learning/environment/ に配置
3. ems_environment.py を修正（下記参照）
4. 事前計算を実行: python station_coverage_calculator.py
5. v13のconfigで学習を実行
"""

import numpy as np
import pandas as pd
import json
import h3
from pathlib import Path
from typing import Dict, Set
import sys

# ==============================================================================
# 1. 事前計算モジュールのテスト
# ==============================================================================

def test_station_coverage_calculator():
    """StationCoverageCalculatorの動作テスト"""
    print("=" * 60)
    print("StationCoverageCalculator テスト")
    print("=" * 60)
    
    # モジュールをインポート
    from station_coverage_calculator import StationCoverageCalculator
    
    # ダミーデータで動作確認
    print("\n1. ダミーデータでの動作テスト")
    
    # 小さな移動時間行列を作成
    n_grids = 100
    travel_time_matrix = np.random.uniform(60, 1200, size=(n_grids, n_grids))
    np.fill_diagonal(travel_time_matrix, 0)  # 同じグリッドへの移動時間は0
    
    # ダミーのgrid_mapping
    grid_mapping = {f"89{i:013x}": i for i in range(n_grids)}
    
    # ダミーの救急車データ
    ambulance_data = pd.DataFrame({
        'latitude': np.random.uniform(35.6, 35.8, 10),
        'longitude': np.random.uniform(139.6, 139.8, 10),
        'team_name': [f'救急隊{i}' for i in range(10)]
    })
    
    # カバレッジ計算器を作成
    calculator = StationCoverageCalculator()
    calculator.compute_coverage(
        ambulance_data=ambulance_data,
        travel_time_matrix=travel_time_matrix,
        grid_mapping=grid_mapping,
        verbose=True
    )
    
    # 統計情報を表示
    stats = calculator.get_statistics()
    print(f"\n統計情報:")
    print(f"  署所数: {stats['num_stations']}")
    print(f"  グリッド数: {stats['num_grids']}")
    
    # カバレッジ損失計算テスト
    print("\n2. カバレッジ損失計算テスト")
    all_station_h3s = set(calculator.station_h3_to_id.keys())
    test_station_h3 = list(all_station_h3s)[0]
    
    # 全署所が利用可能な状態
    loss_6, loss_13 = calculator.calculate_coverage_loss(
        departing_station_h3=test_station_h3,
        available_station_h3s=all_station_h3s
    )
    print(f"  全署所利用可能時の損失: L6={loss_6:.4f}, L13={loss_13:.4f}")
    
    # 1署所のみ利用可能な状態
    loss_6_single, loss_13_single = calculator.calculate_coverage_loss(
        departing_station_h3=test_station_h3,
        available_station_h3s={test_station_h3}
    )
    print(f"  1署所のみ時の損失: L6={loss_6_single:.4f}, L13={loss_13_single:.4f}")
    
    print("\n✓ StationCoverageCalculator テスト完了")
    return calculator


# ==============================================================================
# 2. StateEncoderV2のテスト
# ==============================================================================

def test_state_encoder_v2(calculator):
    """CompactStateEncoderV2の動作テスト"""
    print("\n" + "=" * 60)
    print("CompactStateEncoderV2 テスト")
    print("=" * 60)
    
    from state_encoder_v2 import CompactStateEncoderV2
    
    # 設定
    config = {
        'state_encoding': {
            'mode': 'compact',
            'top_k': 10,
            'normalization': {
                'max_travel_time_minutes': 30,
                'max_station_distance_km': 10
            }
        }
    }
    
    # 小さな移動時間行列
    n_grids = 100
    travel_time_matrix = np.random.uniform(60, 1200, size=(n_grids, n_grids))
    np.fill_diagonal(travel_time_matrix, 0)
    
    grid_mapping = {f"89{i:013x}": i for i in range(n_grids)}
    
    # エンコーダーを作成（新方式）
    encoder = CompactStateEncoderV2(
        config=config,
        top_k=10,
        travel_time_matrix=travel_time_matrix,
        grid_mapping=grid_mapping,
        station_coverage_calculator=calculator
    )
    
    print(f"\n状態次元: {encoder.state_dim}")
    
    # ダミーの状態でエンコードテスト
    state_dict = {
        'ambulances': {
            i: {
                'current_h3': f"89{i:013x}",
                'station_h3': f"89{i:013x}",
                'status': 'available' if i < 8 else 'dispatched'
            }
            for i in range(10)
        },
        'pending_call': {
            'h3_index': f"89{50:013x}",
            'severity': '軽症'
        }
    }
    
    # エンコード
    state_vector = encoder.encode_state(state_dict, grid_mapping)
    print(f"エンコード結果: shape={state_vector.shape}")
    print(f"  候補隊情報（0-39）: min={state_vector[:40].min():.3f}, max={state_vector[:40].max():.3f}")
    print(f"  グローバル状態（40-44）: {state_vector[40:45]}")
    print(f"  傷病度（45）: {state_vector[45]}")
    
    print("\n✓ CompactStateEncoderV2 テスト完了")


# ==============================================================================
# 3. EMSEnvironment統合コード例
# ==============================================================================

EMS_ENVIRONMENT_MODIFICATION = """
# ==============================================================================
# ems_environment.py への修正例
# ==============================================================================

# 1. インポートを追加
from .station_coverage_calculator import StationCoverageCalculator
from .state_encoder_v2 import CompactStateEncoderV2, create_state_encoder_v2

# 2. __init__メソッド内で、state_encoderの初期化部分を修正

# --- 修正前 ---
if self.compact_mode:
    from .state_encoder import CompactStateEncoder
    self.state_encoder = CompactStateEncoder(
        config=self.config,
        top_k=self.top_k,
        travel_time_matrix=response_matrix,
        grid_mapping=self.grid_mapping
    )

# --- 修正後 ---
if self.compact_mode:
    # 状態エンコーディングのバージョンを確認
    encoding_version = state_encoding_config.get('version', 'v1')
    
    if encoding_version == 'v2':
        # v2: 決定論的カバレッジ計算
        coverage_config = state_encoding_config.get('coverage_calculation', {})
        station_coverage_file = coverage_config.get('station_coverage_file', 'station_coverage.json')
        
        # StationCoverageCalculatorを読み込み
        coverage_path = self.base_dir / "processed" / station_coverage_file
        if coverage_path.exists():
            self.station_coverage_calculator = StationCoverageCalculator.load(str(coverage_path))
        else:
            # ファイルがない場合は事前計算を実行
            print(f"警告: {coverage_path} が見つかりません。事前計算を実行します...")
            self.station_coverage_calculator = StationCoverageCalculator()
            self.station_coverage_calculator.compute_coverage(
                ambulance_data=self.ambulance_data,
                travel_time_matrix=response_matrix,
                grid_mapping=self.grid_mapping,
                verbose=True
            )
            self.station_coverage_calculator.save(str(coverage_path))
        
        # CompactStateEncoderV2を使用
        from .state_encoder_v2 import CompactStateEncoderV2
        self.state_encoder = CompactStateEncoderV2(
            config=self.config,
            top_k=self.top_k,
            travel_time_matrix=response_matrix,
            grid_mapping=self.grid_mapping,
            station_coverage_calculator=self.station_coverage_calculator
        )
        print("★ 決定論的カバレッジ計算モード (v2) を使用")
    else:
        # v1: 従来のランダムサンプリング方式
        from .state_encoder import CompactStateEncoder
        self.state_encoder = CompactStateEncoder(
            config=self.config,
            top_k=self.top_k,
            travel_time_matrix=response_matrix,
            grid_mapping=self.grid_mapping
        )
        print("★ 従来のカバレッジ計算モード (v1) を使用")
"""


# ==============================================================================
# 4. 実行手順まとめ
# ==============================================================================

EXECUTION_STEPS = """
# ==============================================================================
# 実行手順
# ==============================================================================

# Step 1: ファイルの配置
cp station_coverage_calculator.py reinforcement_learning/environment/
cp state_encoder_v2.py reinforcement_learning/environment/
cp config_hybrid_unified_v13_deterministic.yaml reinforcement_learning/configs/

# Step 2: 事前計算の実行
cd data/tokyo
python ../../reinforcement_learning/environment/station_coverage_calculator.py \\
    --data-dir . \\
    --output processed/station_coverage.json

# Step 3: ems_environment.pyを修正（上記の修正コードを参照）

# Step 4: 学習の実行
cd reinforcement_learning
python train.py --config configs/config_hybrid_unified_v13_deterministic.yaml

# Step 5: 結果の確認
python evaluate.py --model experiments/hybrid_unified_v13_deterministic/best_model.pth
"""


# ==============================================================================
# メイン実行
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("v13 統合テスト")
    print("=" * 60)
    
    # テスト実行
    calculator = test_station_coverage_calculator()
    test_state_encoder_v2(calculator)
    
    # 修正コード例を表示
    print("\n" + "=" * 60)
    print("EMSEnvironment修正コード例")
    print("=" * 60)
    print(EMS_ENVIRONMENT_MODIFICATION)
    
    # 実行手順を表示
    print("\n" + "=" * 60)
    print("実行手順")
    print("=" * 60)
    print(EXECUTION_STEPS)
