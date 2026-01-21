"""
constants.py
傷病度の統一された定数定義

このファイルは、システム全体で一貫した傷病度の扱いを保証するための
共通定数を定義します。
"""

# ================================================================
# 傷病度の基本定義（軽症から死亡にかけて悪い状態）
# ================================================================
SEVERITY_LEVELS = [
    '軽症',      # 最も軽い
    '中等症',    # 
    '重症',      # 
    '重篤',      # 
    '死亡'       # 最も重い
]

# ================================================================
# 英語マッピング（trainer.pyと同じ）
# ================================================================
SEVERITY_ENGLISH_MAPPING = {
    '軽症': 'mild',
    '中等症': 'moderate', 
    '重症': 'severe',
    '重篤': 'critical',
    '死亡': 'fatal'
}

# ================================================================
# グループ化定義（baseline_comparison.pyと同じ）
# ================================================================
SEVERITY_GROUPS = {
    'severe_conditions': ['重症', '重篤', '死亡'],  # 重症系
    'mild_conditions': ['軽症', '中等症'],          # 軽症系
    'critical_conditions': ['重篤'],                # 最重症
    'moderate_conditions': ['中等症']               # 中等症
}

# ================================================================
# 優先度マッピング（数値が小さいほど緊急度が高い）
# ================================================================
SEVERITY_PRIORITY = {
    '軽症': 5,      # 最も優先度が低い
    '中等症': 4,    
    '重症': 3,      
    '重篤': 2,      
    '死亡': 1       # 最も優先度が高い
}

# ================================================================
# インデックスマッピング（state_encoder.py用）
# ================================================================
SEVERITY_INDICES = {
    '軽症': 4,      # 最も重い状態を0にするため逆順
    '中等症': 3,    
    '重症': 2,      
    '重篤': 1,      
    '死亡': 0       # 最も重い状態
}

# ================================================================
# 重み付け（reward_designer.py用）
# ================================================================
SEVERITY_WEIGHTS = {
    '軽症': 0.2,
    '中等症': 0.4,
    '重症': 1.0,
    '重篤': 1.0,
    '死亡': 1.0
}

# ================================================================
# 時間制限（秒）
# ================================================================
SEVERITY_TIME_LIMITS = {
    '軽症': 1080,    # 18分
    '中等症': 900,   # 15分
    '重症': 360,     # 6分
    '重篤': 360,     # 6分
    '死亡': 360      # 6分
}

# ================================================================
# 色設定（可視化用）
# ================================================================
SEVERITY_COLORS = {
    '軽症': '#90EE90',    # 薄い緑
    '中等症': '#FFD700',  # 黄金
    '重症': '#FFA500',    # オレンジ
    '重篤': '#FF4500',    # 赤オレンジ
    '死亡': '#8B0000'     # 濃い赤
}

# ================================================================
# ユーティリティ関数
# ================================================================
def get_severity_index(severity: str) -> int:
    """傷病度のインデックスを取得"""
    return SEVERITY_INDICES.get(severity, -1)  # 不明な傷病度は-1を返す

def get_severity_priority(severity: str) -> int:
    """傷病度の優先度を取得"""
    return SEVERITY_PRIORITY.get(severity, 10)  # 不明な傷病度は低優先度

def get_severity_english(severity: str) -> str:
    """傷病度の英語表記を取得"""
    return SEVERITY_ENGLISH_MAPPING.get(severity, 'unknown')

def get_severity_weight(severity: str) -> float:
    """傷病度の重みを取得"""
    return SEVERITY_WEIGHTS.get(severity, 0.0)  # 不明な傷病度の重みは0

def get_severity_time_limit(severity: str) -> int:
    """傷病度の時間制限を取得（秒）"""
    return SEVERITY_TIME_LIMITS.get(severity, 780)  # デフォルトは13分

def get_severity_color(severity: str) -> str:
    """傷病度の色を取得"""
    return SEVERITY_COLORS.get(severity, '#808080')  # デフォルトはグレー

def is_severe_condition(severity: str) -> bool:
    """重症系かどうかを判定"""
    return severity in SEVERITY_GROUPS['severe_conditions']

def is_mild_condition(severity: str) -> bool:
    """軽症系かどうかを判定"""
    return severity in SEVERITY_GROUPS['mild_conditions']

def sort_severities_by_level(severities: list) -> list:
    """傷病度リストをレベル順にソート"""
    return [s for s in SEVERITY_LEVELS if s in severities]

def validate_severity(severity: str) -> bool:
    """傷病度の妥当性をチェック"""
    return severity in SEVERITY_LEVELS
