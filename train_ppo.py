"""
train_ppo.py
PPO学習の本格実行スクリプト
"""

import argparse
import yaml
import os
import sys
from datetime import datetime
import torch
import numpy as np
from pathlib import Path
import shutil
import json
import wandb

# プロジェクトパスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reinforcement_learning.environment.ems_environment import EMSEnvironment
from reinforcement_learning.agents.ppo_agent import PPOAgent
from reinforcement_learning.training.trainer import PPOTrainer
from reinforcement_learning.config_utils import load_config_with_inheritance


def setup_directories(experiment_name: str) -> Path:
    """実験用ディレクトリの設定"""
    # 基本ディレクトリ（reinforcement_learning/experiments以下に配置）
    base_dir = Path("reinforcement_learning") / "experiments" / "ppo_training" / experiment_name
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリ
    subdirs = ["checkpoints", "logs", "configs", "visualizations"]
    for subdir in subdirs:
        (base_dir / subdir).mkdir(exist_ok=True)
    
    return base_dir


def save_config(config: dict, output_dir: Path):
    """設定ファイルを保存"""
    config_path = output_dir / "configs" / "config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # JSON形式でも保存（解析しやすい）
    json_path = output_dir / "configs" / "config.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"設定ファイル保存: {config_path}")


def print_training_info(config: dict, experiment_name: str, output_dir: Path):
    """学習情報の表示"""
    print("\n" + "=" * 70)
    print("PPO学習設定")
    print("=" * 70)
    print(f"実験名: {experiment_name}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"デバイス: {config['experiment']['device']}")
    print(f"シード値: {config['experiment']['seed']}")
    
    print("\n【学習設定】")
    print(f"総エピソード数: {config['ppo']['n_episodes']:,}")
    print(f"バッチサイズ: {config['ppo']['batch_size']}")
    print(f"学習率 (Actor): {config['ppo']['learning_rate']['actor']}")
    print(f"学習率 (Critic): {config['ppo']['learning_rate']['critic']}")
    
    print("\n【データ設定】")
    print("学習期間:")
    for period in config['data']['train_periods']:
        print(f"  - {period['start_date']} ～ {period['end_date']}")
    print(f"エピソード長: {config['data']['episode_duration_hours']}時間")
    
    print("\n【傷病度設定】")
    # continuous_paramsのweightが設定されている場合はそれを優先表示
    reward_config = config.get('reward', {})
    core_config = reward_config.get('core', {})
    continuous_params = core_config.get('continuous_params', {})
    
    for category, info in config['severity']['categories'].items():
        conditions = ', '.join(info['conditions'])
        # continuous_paramsのweightを優先（設定されている場合）
        if category in continuous_params and 'weight' in continuous_params[category]:
            weight = continuous_params[category]['weight']
        else:
            weight = info.get('reward_weight', 1.0)
        print(f"  {category}: {conditions} (重み: {weight})")
    
    print("\n【評価設定】")
    print(f"評価間隔: {config['evaluation']['interval']}エピソードごと")
    print(f"評価エピソード数: {config['evaluation']['n_eval_episodes']}")
    
    print("=" * 70 + "\n")


def set_random_seed(seed: int):
    """乱数シードの設定"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"乱数シード設定: {seed}")


def check_gpu():
    """GPU情報の確認"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU検出: {gpu_name}")
        print(f"GPUメモリ: {gpu_memory:.1f}GB")
        return "cuda"
    else:
        print("GPUが利用できません。CPUで実行します。")
        return "cpu"


def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='PPO学習実行')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='設定ファイルのパス')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='実験名（省略時は自動生成）')
    parser.add_argument('--resume', type=str, default=None,
                       help='学習再開用チェックポイントパス')
    parser.add_argument('--device', type=str, default=None,
                       help='使用デバイス (cpu/cuda)')
    parser.add_argument('--debug', action='store_true',
                       help='デバッグモード')
    # ★★★【修正箇所①】★★★
    # ベースライン計測モードを起動する引数を追加
    parser.add_argument('--baseline_mode', type=str, choices=['closest', 'severity_based'], default=None,
                       help='ベースライン計測モードで実行（例: closest）。PPO学習は行わず、指定した戦略の性能を計測します。')
    # ハイブリッドモード対応の引数を追加
    parser.add_argument('--hybrid', action='store_true',
                       help='ハイブリッドモードを有効化')
    parser.add_argument('--experiment', type=str, default=None,
                       help='実験用設定ファイル（experiments/内）')
    
    args = parser.parse_args()
    
    # 設定ファイルの読み込み
    if args.experiment:
        config_path = f'reinforcement_learning/experiments/{args.experiment}'
    else:
        config_path = args.config
    
    print(f"\n設定ファイル読み込み: {config_path}")
    
    # パスの解決（複数のパターンを試す）
    possible_paths = [
        config_path,  # 元のパス
        f"reinforcement_learning/{config_path}",  # reinforcement_learning/ を追加
        f"reinforcement_learning/experiments/{config_path}",  # experiments/ を追加
    ]
    
    # 実験用設定ファイルの場合は追加パターンも試す
    if args.experiment:
        possible_paths.extend([
            args.experiment,  # ファイル名のみ
            f"experiments/{args.experiment}",  # experiments/ のみ
        ])
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            print(f"✓ 設定ファイル発見: {config_path}")
            break
    
    if config_path is None:
        print(f"❌ 設定ファイルが見つかりません:")
        print(f"   試行したパス:")
        for path in possible_paths:
            print(f"     - {path}")
        print(f"   現在のディレクトリ: {os.getcwd()}")
        print(f"   利用可能な設定ファイル:")
        
        # 利用可能な設定ファイルを探して表示
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.yaml') and 'config' in file:
                    print(f"     {os.path.join(root, file)}")
        
        sys.exit(1)
    
    # 設定ファイルの読み込みと継承処理
    config = load_config_with_inheritance(config_path)
    
    # ハイブリッドモードの設定
    is_hybrid = config.get('hybrid_mode', {}).get('enabled', False)
    if args.hybrid:
        if 'hybrid_mode' not in config:
            config['hybrid_mode'] = {}
        config['hybrid_mode']['enabled'] = True
        is_hybrid = True
        print("=" * 60)
        print("ハイブリッドモード: 有効")
        print("- 重症系（重症・重篤・死亡）: 直近隊運用")
        print("- 軽症系（軽症・中等症）: PPO学習")
        if 'reward_weights' in config.get('hybrid_mode', {}):
            print(f"- 報酬バランス: RT {config['hybrid_mode']['reward_weights']['response_time']:.0%}, "
                  f"カバレッジ {config['hybrid_mode']['reward_weights']['coverage']:.0%}, "
                  f"稼働 {config['hybrid_mode']['reward_weights']['workload_balance']:.0%}")
        print("=" * 60)
    
    # デバイスの設定
    if args.device:
        config['experiment']['device'] = args.device
    else:
        detected_device = check_gpu()
        if config['experiment']['device'] == 'cuda' and detected_device == 'cpu':
            print("警告: 設定ではCUDAが指定されていますが、GPUが利用できません。")
            config['experiment']['device'] = 'cpu'
    
    # デバッグモードの設定
    if args.debug:
        print("\n【デバッグモード】")
        config['ppo']['n_episodes'] = 10
        config['evaluation']['interval'] = 5
        config['training']['checkpoint_interval'] = 5
        print("エピソード数を10に制限")
    
    # 実験名の設定
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        # 自動生成: ppo_YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"ppo_{timestamp}"
    
    # ディレクトリの準備
    output_dir = setup_directories(experiment_name)
    
    # 設定の保存
    save_config(config, output_dir)
    
    # 学習情報の表示
    print_training_info(config, experiment_name, output_dir)
    
    # ユーザー確認
    if not args.debug:
        response = input("この設定で学習を開始しますか？ (y/n): ")
        if response.lower() != 'y':
            print("学習をキャンセルしました。")
            return
    
    # wandb初期化（ハイブリッドモード対応）
    if config.get('training', {}).get('logging', {}).get('wandb', False):
        wandb_project = config.get('training', {}).get('logging', {}).get('wandb_project', 'ems_ppo')
        
        wandb.init(
            project=wandb_project,
            name=f"{config.get('experiment', {}).get('name', 'unnamed')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            tags=['hybrid'] if is_hybrid else ['normal']
        )
        print(f"W&B初期化完了: {wandb_project}")
    
    # 乱数シードの設定
    set_random_seed(config['experiment']['seed'])
    
    try:
        # 環境の初期化
        print("\n環境を初期化中...")
        env = EMSEnvironment(config_path)
        
        # 状態・行動次元の取得（常に環境の実際の次元を使用）
        # ★修正: configではなく環境から直接取得
        state_dim = env.state_dim
        action_dim = env.action_dim
        
        print(f"状態空間次元: {state_dim} (環境から取得)")
        print(f"行動空間次元: {action_dim} (環境から取得)")
        
        # エージェントの初期化
        print("\nPPOエージェントを初期化中...")
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config['ppo']
        )
        
        # トレーナーの初期化
        print("\nトレーナーを初期化中...")
        trainer = PPOTrainer(agent, env, config, output_dir)
        
        # チェックポイントからの復帰
        start_episode = 0
        if args.resume:
            print(f"\nチェックポイントから復帰: {args.resume}")
            trainer.load_checkpoint(args.resume)
            # チェックポイントファイル名からエピソード番号を抽出
            import re
            match = re.search(r'checkpoint_ep(\d+)\.pth', args.resume)
            if match:
                start_episode = int(match.group(1))
                print(f"エピソード {start_episode} から再開します")
        
        # 学習の実行
        print("\n学習を開始します...")
        print("-" * 60)
        
        best_reward = trainer.train(start_episode=start_episode)
        
        # モデル保存
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{'hybrid' if is_hybrid else 'normal'}_ppo_{timestamp}.pth"
        model_path = os.path.join(model_dir, model_name)
        
        agent.save(model_path)
        print(f"\nモデルを保存しました: {model_path}")
        
        print("\n" + "=" * 70)
        print("処理完了！")
        print(f"結果保存先: {output_dir}")
        
        # ハイブリッドモードの統計出力
        if is_hybrid and hasattr(trainer, 'hybrid_stats'):
            print("\n" + "=" * 60)
            print("ハイブリッドモード学習結果:")
            
            stats = trainer.hybrid_stats
            if stats.get('severe_rt_history') and len(stats['severe_rt_history']) > 0:
                severe_rt_mean = np.mean(stats['severe_rt_history'])
                print(f"- 重症系平均RT: {severe_rt_mean:.1f}秒 ({severe_rt_mean/60:.1f}分)")
            else:
                print("- 重症系平均RT: データなし")
            
            if stats.get('mild_rt_history') and len(stats['mild_rt_history']) > 0:
                mild_rt_mean = np.mean(stats['mild_rt_history'])
                print(f"- 軽症系平均RT: {mild_rt_mean:.1f}秒 ({mild_rt_mean/60:.1f}分)")
            else:
                print("- 軽症系平均RT: データなし")
            
            if stats.get('coverage_history') and len(stats['coverage_history']) > 0:
                coverage_mean = np.mean(stats['coverage_history'])
                print(f"- 平均カバレッジ: {coverage_mean:.2%}")
            else:
                print("- 平均カバレッジ: データなし")
            
            if 'episodes_with_warning' in stats:
                total_episodes = config['ppo']['n_episodes']
                warning_rate = stats['episodes_with_warning'] / total_episodes
                print(f"- 20分超過エピソード: {stats['episodes_with_warning']}回 ({warning_rate:.1%})")
            else:
                print("- 20分超過エピソード: データなし")
            
            print("=" * 60)
        
        if best_reward is not None:
            print(f"\n最終報酬: {best_reward:.2f}")
        else:
            print(f"\n最終報酬: 計算されませんでした")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n学習を中断しました。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if config.get('training', {}).get('logging', {}).get('wandb', False):
            wandb.finish()


if __name__ == "__main__":
    main()