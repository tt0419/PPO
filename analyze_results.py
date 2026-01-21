"""
analyze_results.py
学習結果の分析と可視化
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import argparse

def load_training_stats(experiment_dir: Path):
    """学習統計の読み込み"""
    stats_path = experiment_dir / "logs" / "training_stats.json"
    
    if not stats_path.exists():
        print(f"統計ファイルが見つかりません: {stats_path}")
        return None
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    return stats

def plot_training_curves(stats: dict, output_dir: Path):
    """学習曲線のプロット"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # エピソード報酬
    axes[0, 0].plot(stats['episode_rewards'])
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True)
    
    # 移動平均報酬（100エピソード）
    if len(stats['episode_rewards']) > 100:
        ma_rewards = pd.Series(stats['episode_rewards']).rolling(100).mean()
        axes[0, 1].plot(ma_rewards)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward (MA100)')
        axes[0, 1].set_title('Moving Average Rewards')
        axes[0, 1].grid(True)
    
    # 損失関数
    if 'training_stats' in stats and stats['training_stats']:
        actor_losses = [s.get('actor_loss', 0) for s in stats['training_stats']]
        critic_losses = [s.get('critic_loss', 0) for s in stats['training_stats']]
        
        axes[1, 0].plot(actor_losses, label='Actor', alpha=0.7)
        axes[1, 0].plot(critic_losses, label='Critic', alpha=0.7)
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 評価報酬
    if 'eval_stats' in stats and stats['eval_stats']:
        eval_rewards = [s['mean_reward'] for s in stats['eval_stats']]
        eval_episodes = list(range(0, len(eval_rewards) * 100, 100))
        axes[1, 1].plot(eval_episodes, eval_rewards, marker='o')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Eval Reward')
        axes[1, 1].set_title('Evaluation Performance')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.show()
    
    print(f"グラフ保存: {output_dir / 'training_curves.png'}")

def analyze_performance(stats: dict):
    """性能分析"""
    print("\n" + "=" * 60)
    print("学習結果分析")
    print("=" * 60)
    
    # 基本統計
    rewards = stats['episode_rewards']
    print(f"\n【エピソード報酬統計】")
    print(f"総エピソード数: {len(rewards)}")
    print(f"最大報酬: {max(rewards):.2f}")
    print(f"最小報酬: {min(rewards):.2f}")
    print(f"平均報酬: {np.mean(rewards):.2f}")
    print(f"標準偏差: {np.std(rewards):.2f}")
    
    # 最後100エピソードの統計
    if len(rewards) > 100:
        recent_rewards = rewards[-100:]
        print(f"\n【最後100エピソードの統計】")
        print(f"平均報酬: {np.mean(recent_rewards):.2f}")
        print(f"最大報酬: {max(recent_rewards):.2f}")
        print(f"最小報酬: {min(recent_rewards):.2f}")
    
    # 評価結果
    if 'eval_stats' in stats and stats['eval_stats']:
        print(f"\n【評価結果】")
        eval_rewards = [s['mean_reward'] for s in stats['eval_stats']]
        print(f"評価回数: {len(eval_rewards)}")
        print(f"最良評価報酬: {max(eval_rewards):.2f}")
        print(f"最終評価報酬: {eval_rewards[-1]:.2f}")

def main():
    parser = argparse.ArgumentParser(description='学習結果の分析')
    parser.add_argument('experiment_dir', type=str,
                       help='実験ディレクトリのパス')
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"ディレクトリが見つかりません: {experiment_dir}")
        return
    
    # 統計の読み込み
    stats = load_training_stats(experiment_dir)
    if stats is None:
        return
    
    # 分析実行
    analyze_performance(stats)
    
    # 可視化
    plot_training_curves(stats, experiment_dir)

if __name__ == "__main__":
    main()