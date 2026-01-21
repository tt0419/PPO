"""
modular_state_encoder.py
特徴量タイプごとに専用のエンコーダを持つ改良版StateEncoder
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class ModularStateEncoder(nn.Module):
    """特徴量タイプごとに専用のエンコーダを持つアーキテクチャ"""
    
    def __init__(self, max_ambulances: int = 16):
        super().__init__()
        self.max_ambulances = max_ambulances
        
        # 各特徴量の次元
        self.ambulance_dim = max_ambulances * 5  # 16*5 = 80
        self.incident_dim = 10
        self.temporal_dim = 8
        self.spatial_dim = 21
        
        # 1. 救急車エンコーダ（80次元 → 64次元）
        self.ambulance_encoder = nn.Sequential(
            nn.Linear(self.ambulance_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 2. 事案エンコーダ（10次元 → 32次元）
        self.incident_encoder = nn.Sequential(
            nn.Linear(self.incident_dim, 24),
            nn.LayerNorm(24),
            nn.ReLU(),
            nn.Linear(24, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # 3. 時間エンコーダ（8次元 → 16次元）
        self.temporal_encoder = nn.Sequential(
            nn.Linear(self.temporal_dim, 12),
            nn.LayerNorm(12),
            nn.ReLU(),
            nn.Linear(12, 16)
        )
        
        # 4. 空間統計エンコーダ（21次元 → 32次元）
        self.spatial_encoder = nn.Sequential(
            nn.Linear(self.spatial_dim, 28),
            nn.LayerNorm(28),
            nn.ReLU(),
            nn.Linear(28, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # 5. 特徴量の重要度を学習するアテンション機構
        total_encoded_dim = 64 + 32 + 16 + 32  # = 144
        self.feature_attention = nn.Sequential(
            nn.Linear(total_encoded_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4つの特徴量グループ
            nn.Softmax(dim=-1)
        )
        
        # 6. 最終的な特徴量統合層（144次元 → 128次元）
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_encoded_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU()
        )
        
    def forward(self, state):
        """
        state: [batch_size, state_dim] のテンソル
        state_dim = 80 + 10 + 8 + 21 = 119次元
        """
        # 特徴量を分割
        idx = 0
        ambulance_features = state[:, idx:idx+self.ambulance_dim]
        idx += self.ambulance_dim
        
        incident_features = state[:, idx:idx+self.incident_dim]
        idx += self.incident_dim
        
        temporal_features = state[:, idx:idx+self.temporal_dim]
        idx += self.temporal_dim
        
        spatial_features = state[:, idx:idx+self.spatial_dim]
        
        # 各特徴量を個別にエンコード
        ambulance_encoded = self.ambulance_encoder(ambulance_features)
        incident_encoded = self.incident_encoder(incident_features)
        temporal_encoded = self.temporal_encoder(temporal_features)
        spatial_encoded = self.spatial_encoder(spatial_features)
        
        # 特徴量を結合
        combined = torch.cat([
            ambulance_encoded,
            incident_encoded,
            temporal_encoded,
            spatial_encoded
        ], dim=-1)
        
        # アテンション重みを計算
        attention_weights = self.feature_attention(combined)
        
        # 重み付き結合（各特徴量に対応する重みを適用）
        weighted_ambulance = ambulance_encoded * attention_weights[:, 0:1]
        weighted_incident = incident_encoded * attention_weights[:, 1:2]
        weighted_temporal = temporal_encoded * attention_weights[:, 2:3]
        weighted_spatial = spatial_encoded * attention_weights[:, 3:4]
        
        weighted_features = torch.cat([
            weighted_ambulance,
            weighted_incident,
            weighted_temporal,
            weighted_spatial
        ], dim=-1)
        
        # 最終的な特徴量を生成
        output = self.fusion_layer(weighted_features)
        return output

    @property
    def output_dim(self):
        """エンコード後の出力次元数"""
        return 96