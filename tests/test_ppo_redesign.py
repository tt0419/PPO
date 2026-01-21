"""
test_ppo_redesign.py
PPO環境再設計のテスト

設計仕様書（ppo_redesign_specification.md）に基づくテスト
"""

import numpy as np
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_state_encoding():
    """状態エンコーディングのテスト（46次元）"""
    print("\n" + "=" * 60)
    print("テスト1: 状態エンコーディング（46次元）")
    print("=" * 60)
    
    from reinforcement_learning.environment.state_encoder import CompactStateEncoder
    
    # テスト用設定
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
    
    encoder = CompactStateEncoder(config, top_k=10)
    
    # 1. 次元数の確認
    expected_dim = 46  # 40 (候補隊) + 5 (グローバル) + 1 (傷病度)
    print(f"  期待する次元数: {expected_dim}")
    print(f"  実際の次元数: {encoder.state_dim}")
    assert encoder.state_dim == expected_dim, f"期待: {expected_dim}, 実際: {encoder.state_dim}"
    print("  ✓ 次元数チェック通過")
    
    # 2. ダミーデータで状態を生成
    dummy_ambulances = {}
    for i in range(10):
        dummy_ambulances[i] = {
            'current_h3': '892a100d2c3ffff',  # ダミーH3
            'station_h3': '892a100d2c3ffff',
            'status': 'available' if i < 8 else 'dispatched',
            'calls_today': i
        }
    
    dummy_incident = {
        'h3_index': '892a100d2c3ffff',
        'severity': '軽症'
    }
    
    state_dict = {
        'ambulances': dummy_ambulances,
        'pending_call': dummy_incident,
        'time_of_day': 12.0
    }
    
    # grid_mappingなしでエンコード（ダミーデータが返される）
    state = encoder.encode_state(state_dict, grid_mapping=None)
    
    print(f"  状態ベクトル形状: {state.shape}")
    assert state.shape[0] == 46, f"期待: 46, 実際: {state.shape[0]}"
    print("  ✓ 形状チェック通過")
    
    # 3. 値の範囲確認（0-1に正規化されているか）
    assert np.all(state >= 0) and np.all(state <= 1), "状態値が[0, 1]の範囲外"
    print("  ✓ 値範囲チェック通過")
    
    # 4. 構造の確認
    print("\n  状態ベクトル構造:")
    for i in range(10):
        base_idx = i * 4
        print(f"    候補隊{i+1}: 移動時間={state[base_idx]:.3f}, "
              f"移動距離={state[base_idx+1]:.3f}, "
              f"L6={state[base_idx+2]:.3f}, L13={state[base_idx+3]:.3f}")
    
    print(f"    グローバル[40]: 利用可能率={state[40]:.3f}")
    print(f"    グローバル[41]: 出場中率={state[41]:.3f}")
    print(f"    グローバル[42]: 6分圏内率={state[42]:.3f}")
    print(f"    グローバル[43]: 平均移動時間={state[43]:.3f}")
    print(f"    グローバル[44]: システムC6={state[44]:.3f}")
    print(f"    傷病度[45]: {state[45]:.3f} (0=重症系, 1=軽症系)")
    
    print("\n  ✓ 状態エンコーディングテスト通過")
    return True


def test_reward_calculation():
    """報酬計算のテスト"""
    print("\n" + "=" * 60)
    print("テスト2: 報酬計算")
    print("=" * 60)
    
    from reinforcement_learning.environment.reward_designer import RewardDesigner
    
    # テスト用設定
    config = {
        'reward': {
            'unified': {
                'critical_max_bonus': 50.0,
                'critical_lambda': 0.115,
                'critical_penalty_scale': 5.0,
                'critical_penalty_power': 1.5,
                'mild_max_bonus': 10.0,
                'mild_penalty_scale': 1.0,
                'coverage_w6': 0.5,
                'coverage_w13': 0.5,
                'coverage_penalty_scale': 10.0,
                'time_weight': 0.6,
                'coverage_weight': 0.4,
            },
            'system': {
                'dispatch_failure': -1.0,
                'no_available_ambulance': 0.0
            }
        },
        'hybrid_mode': {'enabled': False}
    }
    
    reward_designer = RewardDesigner(config)
    
    # テスト1: 軽症系、目標時間内、カバレッジ損失小
    print("\n  テスト1: 軽症系、10分、L6=0.1, L13=0.05")
    reward1 = reward_designer.calculate_step_reward(
        severity='軽症',
        response_time_sec=600,  # 10分
        L6=0.1,
        L13=0.05
    )
    print(f"    報酬: {reward1:.2f}")
    assert reward1 > 0, "目標時間内・カバレッジ損失小で報酬が正であるべき"
    print("    ✓ 通過")
    
    # テスト2: 軽症系、目標時間超過
    print("\n  テスト2: 軽症系、20分、L6=0.1, L13=0.05")
    reward2 = reward_designer.calculate_step_reward(
        severity='軽症',
        response_time_sec=1200,  # 20分
        L6=0.1,
        L13=0.05
    )
    print(f"    報酬: {reward2:.2f}")
    assert reward2 < reward1, "超過時は報酬が減少すべき"
    print("    ✓ 通過")
    
    # テスト3: 軽症系、カバレッジ損失大
    print("\n  テスト3: 軽症系、10分、L6=0.8, L13=0.7")
    reward3 = reward_designer.calculate_step_reward(
        severity='軽症',
        response_time_sec=600,  # 10分
        L6=0.8,
        L13=0.7
    )
    print(f"    報酬: {reward3:.2f}")
    assert reward3 < reward1, "カバレッジ損失大で報酬が減少すべき"
    print("    ✓ 通過")
    
    # テスト4: 重症系（通常PPOモード）
    print("\n  テスト4: 重症系、5分、通常PPO")
    reward4 = reward_designer.calculate_step_reward(
        severity='重症',
        response_time_sec=300,  # 5分
        L6=0.1,
        L13=0.05
    )
    print(f"    報酬: {reward4:.2f}")
    assert reward4 > 0, "重症系・目標時間内で報酬が正であるべき"
    print("    ✓ 通過")
    
    # テスト5: 重症系、目標超過
    print("\n  テスト5: 重症系、10分、通常PPO（目標超過）")
    reward5 = reward_designer.calculate_step_reward(
        severity='重症',
        response_time_sec=600,  # 10分
        L6=0.1,
        L13=0.05
    )
    print(f"    報酬: {reward5:.2f}")
    assert reward5 < reward4, "重症系で超過するとペナルティ"
    print("    ✓ 通過")
    
    # テスト6: 重症系（ハイブリッドモード）
    print("\n  テスト6: 重症系、5分、ハイブリッドPPO")
    config['hybrid_mode']['enabled'] = True
    reward_designer_hybrid = RewardDesigner(config)
    reward6 = reward_designer_hybrid.calculate_step_reward(
        severity='重症',
        response_time_sec=300,
        L6=0.1,
        L13=0.05
    )
    print(f"    報酬: {reward6:.2f}")
    assert reward6 == 0.0, "ハイブリッドモードの重症系は報酬0"
    print("    ✓ 通過")
    
    print("\n  ✓ 報酬計算テスト通過")
    return True


def test_coverage_loss_calculation():
    """カバレッジ損失計算のテスト"""
    print("\n" + "=" * 60)
    print("テスト3: カバレッジ損失計算")
    print("=" * 60)
    
    from reinforcement_learning.environment.state_encoder import CompactStateEncoder
    
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
    
    encoder = CompactStateEncoder(config, top_k=10)
    
    # ダミーの救急隊リスト
    all_available = [
        {'id': 0, 'current_h3': '892a100d2c3ffff', 'station_h3': '892a100d2c3ffff'},
        {'id': 1, 'current_h3': '892a100d2c3ffff', 'station_h3': '892a100d2c3ffff'},
        {'id': 2, 'current_h3': '892a100d2c3ffff', 'station_h3': '892a100d2c3ffff'},
    ]
    
    ambulance = all_available[0]
    
    # travel_time_matrixとgrid_mappingがないのでデフォルト値が返される
    L6, L13 = encoder.calculate_coverage_loss(
        ambulance, all_available, None, None
    )
    
    print(f"  L6 (6分カバレッジ損失): {L6:.3f}")
    print(f"  L13 (13分カバレッジ損失): {L13:.3f}")
    
    assert 0 <= L6 <= 1, f"L6が範囲外: {L6}"
    assert 0 <= L13 <= 1, f"L13が範囲外: {L13}"
    print("  ✓ 値範囲チェック通過")
    
    # 他に救急車がない場合
    L6_max, L13_max = encoder.calculate_coverage_loss(
        ambulance, [ambulance], None, None
    )
    print(f"\n  他に救急車がない場合:")
    print(f"    L6: {L6_max:.3f}, L13: {L13_max:.3f}")
    assert L6_max == 1.0 and L13_max == 1.0, "他に救急車がない場合は最大損失"
    print("  ✓ 最大損失チェック通過")
    
    print("\n  ✓ カバレッジ損失計算テスト通過")
    return True


def test_reward_info():
    """報酬設計情報取得のテスト"""
    print("\n" + "=" * 60)
    print("テスト4: 報酬設計情報取得")
    print("=" * 60)
    
    from reinforcement_learning.environment.reward_designer import RewardDesigner
    
    config = {
        'reward': {
            'unified': {
                'time_weight': 0.6,
                'coverage_weight': 0.4,
            },
            'system': {}
        },
        'hybrid_mode': {'enabled': False}
    }
    
    reward_designer = RewardDesigner(config)
    info = reward_designer.get_info()
    
    print(f"  ハイブリッドモード: {info['hybrid_mode']}")
    print(f"  時間重み: {info['time_weight']}")
    print(f"  カバレッジ重み: {info['coverage_weight']}")
    print(f"  重症系パラメータ: {info['critical_params']}")
    print(f"  軽症系パラメータ: {info['mild_params']}")
    print(f"  カバレッジパラメータ: {info['coverage_params']}")
    
    assert 'hybrid_mode' in info
    assert 'time_weight' in info
    assert 'coverage_weight' in info
    assert 'critical_params' in info
    assert 'mild_params' in info
    assert 'coverage_params' in info
    
    print("\n  ✓ 報酬設計情報取得テスト通過")
    return True


def run_all_tests():
    """全テストを実行"""
    print("\n" + "=" * 70)
    print("PPO環境再設計テスト")
    print("=" * 70)
    
    results = []
    
    try:
        results.append(("状態エンコーディング", test_state_encoding()))
    except Exception as e:
        print(f"  ✗ 状態エンコーディングテスト失敗: {e}")
        results.append(("状態エンコーディング", False))
    
    try:
        results.append(("報酬計算", test_reward_calculation()))
    except Exception as e:
        print(f"  ✗ 報酬計算テスト失敗: {e}")
        results.append(("報酬計算", False))
    
    try:
        results.append(("カバレッジ損失計算", test_coverage_loss_calculation()))
    except Exception as e:
        print(f"  ✗ カバレッジ損失計算テスト失敗: {e}")
        results.append(("カバレッジ損失計算", False))
    
    try:
        results.append(("報酬設計情報取得", test_reward_info()))
    except Exception as e:
        print(f"  ✗ 報酬設計情報取得テスト失敗: {e}")
        results.append(("報酬設計情報取得", False))
    
    # 結果サマリー
    print("\n" + "=" * 70)
    print("テスト結果サマリー")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ 通過" if success else "✗ 失敗"
        print(f"  {name}: {status}")
    
    print(f"\n  合計: {passed}/{total} 通過")
    
    if passed == total:
        print("\n  ✓ すべてのテストが通過しました！")
    else:
        print("\n  ✗ 一部のテストが失敗しました")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
