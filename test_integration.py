#!/usr/bin/env python
"""
test_integration.py
46次元PPO環境の統合テスト

このスクリプトは以下を確認します：
1. 環境の初期化
2. 状態空間の次元
3. エージェントの初期化
4. 1エピソードの実行
5. 報酬計算の動作
"""

import sys
import os
import numpy as np

# パス設定
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_environment_initialization(config_path: str):
    """環境初期化テスト"""
    print("\n" + "=" * 60)
    print("テスト1: 環境初期化")
    print("=" * 60)
    
    from reinforcement_learning.environment.ems_environment import EMSEnvironment
    
    env = EMSEnvironment(config_path, mode='train')
    
    print(f"\n✓ 環境初期化成功")
    print(f"  状態次元: {env.state_dim}")
    print(f"  行動次元: {env.action_dim}")
    print(f"  コンパクトモード: {env.compact_mode}")
    print(f"  Top-K: {env.top_k}")
    
    # 次元の確認
    assert env.state_dim == 46, f"状態次元が46ではありません: {env.state_dim}"
    assert env.action_dim == 10, f"行動次元が10ではありません: {env.action_dim}"
    
    print("\n✓ 次元チェック通過")
    return env


def test_agent_initialization(env, config_path: str):
    """エージェント初期化テスト"""
    print("\n" + "=" * 60)
    print("テスト2: エージェント初期化")
    print("=" * 60)
    
    from reinforcement_learning.agents.ppo_agent import PPOAgent
    from reinforcement_learning.config_utils import load_config_with_inheritance
    
    config = load_config_with_inheritance(config_path)
    
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config=config['ppo']
    )
    
    print(f"\n✓ エージェント初期化成功")
    print(f"  Actor入力次元: {agent.actor.state_dim}")
    print(f"  Actor出力次元: {agent.actor.action_dim}")
    
    # ModularEncoderが無効化されているか確認
    if hasattr(agent.actor, 'use_modular_encoder'):
        print(f"  ModularEncoder使用: {agent.actor.use_modular_encoder}")
        assert not agent.actor.use_modular_encoder, "ModularEncoderが有効になっています"
        print("  ✓ ModularEncoder無効化確認")
    
    return agent


def test_episode_execution(env, agent, max_steps: int = 50):
    """エピソード実行テスト"""
    print("\n" + "=" * 60)
    print("テスト3: エピソード実行")
    print("=" * 60)
    
    state = env.reset()
    
    print(f"\n初期状態:")
    print(f"  形状: {state.shape}")
    print(f"  値範囲: [{state.min():.3f}, {state.max():.3f}]")
    
    assert state.shape[0] == 46, f"状態の形状が46ではありません: {state.shape}"
    
    total_reward = 0.0
    rewards = []
    response_times = []
    coverage_losses = []
    
    print(f"\nエピソード実行中（最大{max_steps}ステップ）...")
    
    for step in range(max_steps):
        # アクションマスク取得
        mask = env.get_action_mask()
        
        if mask.sum() == 0:
            print(f"  ステップ{step}: 利用可能な救急車なし、スキップ")
            continue
        
        # 行動選択
        action, log_prob, value = agent.select_action(state, mask)
        
        # ステップ実行
        result = env.step(action)
        
        if result is None:
            print(f"  ステップ{step}: resultがNone")
            break
        
        # 報酬記録
        rewards.append(result.reward)
        total_reward += result.reward
        
        # 詳細情報の記録
        if 'dispatch_result' in result.info:
            dispatch = result.info['dispatch_result']
            if dispatch.get('success'):
                rt_min = dispatch.get('response_time_minutes', 0)
                response_times.append(rt_min)
                
                L6 = dispatch.get('coverage_loss_6min', 0)
                L13 = dispatch.get('coverage_loss_13min', 0)
                coverage_losses.append((L6, L13))
        
        # 最初の5ステップは詳細表示
        if step < 5:
            severity = result.info.get('severity', 'unknown')
            rt = result.info.get('dispatch_result', {}).get('response_time_minutes', 0)
            print(f"  ステップ{step}: action={action}, reward={result.reward:.2f}, "
                  f"severity={severity}, RT={rt:.1f}分")
        
        state = result.observation
        
        if result.done:
            print(f"\n  エピソード終了（ステップ{step+1}）")
            break
    
    # 結果サマリー
    print(f"\n結果サマリー:")
    print(f"  総ステップ数: {step+1}")
    print(f"  累積報酬: {total_reward:.2f}")
    
    if rewards:
        print(f"  報酬統計: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}")
        print(f"  報酬範囲: [{min(rewards):.2f}, {max(rewards):.2f}]")
    
    if response_times:
        print(f"  応答時間統計: mean={np.mean(response_times):.1f}分, "
              f"median={np.median(response_times):.1f}分")
        rate_6min = sum(1 for rt in response_times if rt <= 6) / len(response_times) * 100
        rate_13min = sum(1 for rt in response_times if rt <= 13) / len(response_times) * 100
        print(f"  6分達成率: {rate_6min:.1f}%, 13分達成率: {rate_13min:.1f}%")
    
    if coverage_losses:
        L6_mean = np.mean([l[0] for l in coverage_losses])
        L13_mean = np.mean([l[1] for l in coverage_losses])
        print(f"  カバレッジ損失: L6={L6_mean:.3f}, L13={L13_mean:.3f}")
        
        # L6, L13が0.5以外の値を取っているか確認
        if L6_mean != 0.5 or L13_mean != 0.5:
            print(f"  ✓ カバレッジ損失が正しく計算されています")
        else:
            print(f"  ⚠️ カバレッジ損失がデフォルト値（0.5）のままです")
    
    print("\n✓ エピソード実行テスト通過")
    return total_reward


def test_reward_designer(env):
    """報酬設計テスト"""
    print("\n" + "=" * 60)
    print("テスト4: 報酬設計確認")
    print("=" * 60)
    
    rd = env.reward_designer
    
    print(f"\n報酬設計情報:")
    info = rd.get_info()
    print(f"  ハイブリッドモード: {info['hybrid_mode']}")
    print(f"  時間重み: {info['time_weight']}")
    print(f"  カバレッジ重み: {info['coverage_weight']}")
    
    # テストケース
    print(f"\n報酬計算テスト:")
    
    # 軽症、目標内
    r1 = rd.calculate_step_reward('軽症', 600, L6=0.1, L13=0.05)
    print(f"  軽症, 10分, L6=0.1, L13=0.05 → 報酬: {r1:.2f}")
    
    # 軽症、目標超過
    r2 = rd.calculate_step_reward('軽症', 1200, L6=0.1, L13=0.05)
    print(f"  軽症, 20分, L6=0.1, L13=0.05 → 報酬: {r2:.2f}")
    
    # 重症（ハイブリッドモードなら0）
    r3 = rd.calculate_step_reward('重症', 300, L6=0.1, L13=0.05)
    print(f"  重症, 5分, L6=0.1, L13=0.05 → 報酬: {r3:.2f}")
    
    if info['hybrid_mode']:
        assert r3 == 0.0, "ハイブリッドモードで重症系の報酬が0ではありません"
        print(f"  ✓ ハイブリッドモード: 重症系報酬=0 確認")
    
    print("\n✓ 報酬設計テスト通過")


def run_integration_test(config_path: str):
    """統合テスト実行"""
    print("\n" + "=" * 70)
    print("PPO環境再設計 統合テスト")
    print("=" * 70)
    print(f"設定ファイル: {config_path}")
    
    try:
        # テスト1: 環境初期化
        env = test_environment_initialization(config_path)
        
        # テスト2: エージェント初期化
        agent = test_agent_initialization(env, config_path)
        
        # テスト3: エピソード実行
        total_reward = test_episode_execution(env, agent, max_steps=50)
        
        # テスト4: 報酬設計確認
        test_reward_designer(env)
        
        # 最終結果
        print("\n" + "=" * 70)
        print("統合テスト結果: すべて通過")
        print("=" * 70)
        print("\n次のステップ:")
        print("  1. 短期学習テスト（100エピソード）を実行")
        print("     python train_ppo.py --config config_hybrid_short_test.yaml --debug")
        print("  2. 学習曲線を確認")
        print("  3. 本格学習を開始")
        
        return True
        
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO環境統合テスト')
    parser.add_argument('--config', type=str, 
                        default='config_hybrid_short_test.yaml',
                        help='設定ファイルのパス')
    
    args = parser.parse_args()
    
    success = run_integration_test(args.config)
    sys.exit(0 if success else 1)
