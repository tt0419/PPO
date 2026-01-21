"""
network_architectures.py
Actor-Criticネットワークの定義
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

# ModularStateEncoderのインポート
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment.modular_state_encoder import ModularStateEncoder

class ActorNetwork(nn.Module):
    """
    改良版Actor Network（Modular State Encoderを使用）
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # ★★★ ModularStateEncoderのデフォルトをFalseに変更 ★★★
        self.use_modular_encoder = config.get('use_modular_encoder', False)
        
        if self.use_modular_encoder:
            # 設定から救急車数を取得（デフォルト16）
            num_ambulances = config.get('num_ambulances', 16)
            self.state_encoder = ModularStateEncoder(max_ambulances=num_ambulances)
            encoded_dim = self.state_encoder.output_dim  # 96次元
        else:
            # 従来の方法（フォールバック）
            encoded_dim = state_dim
            self.state_encoder = None
        
        # ポリシーネットワーク
        hidden_layers = config.get('network', {}).get('actor', {}).get('hidden_layers', [128, 64])
        activation = config.get('network', {}).get('actor', {}).get('activation', 'relu')
        dropout_rate = config.get('network', {}).get('actor', {}).get('dropout', 0.1)
        
        # 活性化関数の選択
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # ネットワーク層の構築
        layers = []
        prev_dim = encoded_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 最終層（行動確率出力）
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.policy_network = nn.Sequential(*layers)
        
        # 重み初期化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """重みの初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            state: 状態テンソル [batch_size, state_dim]
            
        Returns:
            action_probs: 行動確率 [batch_size, action_dim]
        """
        # ★★★ Modular Encoderを使用 ★★★
        if self.state_encoder is not None:
            encoded_state = self.state_encoder(state)
        else:
            encoded_state = state
        
        # ポリシーネットワークを通す
        logits = self.policy_network(encoded_state)
        
        # Softmaxで確率分布に変換
        action_probs = F.softmax(logits, dim=-1)
        
        # 数値安定性のため小さな値を加える
        action_probs = action_probs + 1e-8
        
        return action_probs


class CriticNetwork(nn.Module):
    """
    改良版Critic Network（Modular State Encoderを使用）
    """
    
    def __init__(self, state_dim: int, config: Dict):
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # ★★★ ModularStateEncoderのデフォルトをFalseに変更 ★★★
        self.use_modular_encoder = config.get('use_modular_encoder', False)
        
        if self.use_modular_encoder:
            num_ambulances = config.get('num_ambulances', 16)
            self.state_encoder = ModularStateEncoder(max_ambulances=num_ambulances)
            encoded_dim = self.state_encoder.output_dim  # 96次元
        else:
            encoded_dim = state_dim
            self.state_encoder = None
        
        # 価値ネットワーク
        hidden_layers = config.get('network', {}).get('critic', {}).get('hidden_layers', [128, 64])
        activation = config.get('network', {}).get('critic', {}).get('activation', 'relu')
        dropout_rate = config.get('network', {}).get('critic', {}).get('dropout', 0.1)
        
        # 活性化関数
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ReLU()
        
        # ネットワーク層の構築
        layers = []
        prev_dim = encoded_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)
        
        # 重み初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みの初期化"""
        for module in self.feature_layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.constant_(self.value_head.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """順伝播"""
        # ★★★ Modular Encoderを使用 ★★★
        if self.state_encoder is not None:
            encoded_state = self.state_encoder(state)
        else:
            encoded_state = state
        
        features = self.feature_layers(encoded_state)
        value = self.value_head(features)
        return value


class AttentionActorNetwork(nn.Module):
    """
    注意機構を持つActorネットワーク（発展版）
    救急車と事案の関係を学習
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        super(AttentionActorNetwork, self).__init__()
        
        self.action_dim = action_dim  # 192
        self.num_ambulances = 192
        self.ambulance_features = 4
        self.incident_features = 10
        
        # 救急車エンコーダ
        self.ambulance_encoder = nn.Sequential(
            nn.Linear(self.ambulance_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 事案エンコーダ
        self.incident_encoder = nn.Sequential(
            nn.Linear(self.incident_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # 注意機構
        self.attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )
        
        # 最終層
        self.output_layer = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        注意機構を使った順伝播
        """
        batch_size = state.shape[0]
        
        # 状態を分解
        # 救急車情報: [batch_size, num_ambulances, ambulance_features]
        ambulance_states = state[:, :self.num_ambulances * self.ambulance_features]
        ambulance_states = ambulance_states.view(batch_size, self.num_ambulances, self.ambulance_features)
        
        # 事案情報: [batch_size, incident_features]
        incident_state = state[:, self.num_ambulances * self.ambulance_features:
                               self.num_ambulances * self.ambulance_features + self.incident_features]
        
        # エンコード
        ambulance_encoded = self.ambulance_encoder(ambulance_states)  # [batch, 192, 32]
        incident_encoded = self.incident_encoder(incident_state)  # [batch, 32]
        incident_encoded = incident_encoded.unsqueeze(1)  # [batch, 1, 32]
        
        # 注意機構（事案を基準に救急車を評価）
        attended, _ = self.attention(
            query=incident_encoded,
            key=ambulance_encoded,
            value=ambulance_encoded
        )  # [batch, 1, 32]
        
        # 各救急車に対する相対スコア
        scores = self.output_layer(ambulance_encoded)  # [batch, 192, 1]
        scores = scores.squeeze(-1)  # [batch, 192]
        
        # Softmaxで確率分布に
        action_probs = F.softmax(scores, dim=-1)
        
        return action_probs