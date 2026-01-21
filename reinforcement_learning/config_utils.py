"""
config_utils.py
設定ファイル処理のユーティリティ
"""

import yaml
import os
from pathlib import Path


def deep_merge_config(base_config: dict, override_config: dict) -> dict:
    """
    設定辞書の深いマージ
    override_configの値でbase_configを上書き
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_config(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config_with_inheritance(config_path: str) -> dict:
    """
    設定ファイルの継承機能付き読み込み
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        マージされた設定辞書
    """
    # 指定された設定ファイルを読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        specific_config = yaml.safe_load(f)
    
    # ベース設定ファイル (config.yaml) のパスを検索
    base_config_path = None
    possible_base_paths = [
        "config.yaml",
        "reinforcement_learning/experiments/config.yaml", 
        Path(config_path).parent / "config.yaml"
    ]
    
    for path in possible_base_paths:
        if os.path.exists(path):
            base_config_path = path
            break
    
    # ベース設定が見つからない場合は、指定された設定のみを使用
    if base_config_path is None:
        print("注意: ベース設定ファイル (config.yaml) が見つかりません。指定された設定のみを使用します。")
        return specific_config
    
    # ベース設定を読み込み
    print(f"ベース設定読み込み: {base_config_path}")
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # 設定をマージ（specific_configでbase_configを上書き）
    merged_config = deep_merge_config(base_config, specific_config)
    
    # continuous_paramsのweightをseverity.categories.reward_weightに自動同期
    # （設定の一貫性を保つため）
    _sync_continuous_params_weight(merged_config)
    
    print("✓ 設定ファイル継承完了")
    return merged_config


def _sync_continuous_params_weight(config: dict):
    """
    continuous_paramsのweightをseverity.categories.reward_weightに自動同期
    
    continuous_params.weightが設定されている場合、severity.categories.reward_weightを
    自動的に更新して設定の一貫性を保つ
    """
    reward_config = config.get('reward', {})
    core_config = reward_config.get('core', {})
    continuous_params = core_config.get('continuous_params', {})
    
    # continuous_paramsにweightが設定されている場合のみ同期
    if not continuous_params:
        return
    
    severity_config = config.get('severity', {})
    categories = severity_config.get('categories', {})
    
    # 各カテゴリのweightを同期
    for category in ['critical', 'moderate', 'mild']:
        if category in continuous_params and 'weight' in continuous_params[category]:
            weight = continuous_params[category]['weight']
            if category in categories:
                categories[category]['reward_weight'] = weight