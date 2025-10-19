"""Emotion impact prediction module.

Estimates emotional responses and impacts of actions on self and others.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import EmotionalImpact


class EmotionDeltaPredictor(nn.Module):
    """Predicts changes in emotional state."""
    
    def __init__(self, hidden_dim: int = 768, emotion_dim: int = 256):
        super().__init__()
        
        self.delta_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, emotion_dim),
            nn.Tanh(),  # Emotion deltas in [-1, 1]
        )
        
    def forward(self, action_context: torch.Tensor) -> torch.Tensor:
        """Predict emotion delta from action context."""
        return self.delta_net(action_context)


class ValenceArousalPredictor(nn.Module):
    """Predicts valence (positive/negative) and arousal (intensity) of emotions."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.valence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Valence in [-1, 1]
        )
        
        self.arousal_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Arousal in [0, 1]
        )
        
    def forward(self, action_context: torch.Tensor) -> tuple:
        """Predict valence and arousal."""
        valence = self.valence_net(action_context).squeeze(-1)
        arousal = self.arousal_net(action_context).squeeze(-1)
        return valence, arousal


class EmotionImpactModule(nn.Module):
    """Complete emotion impact prediction module."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        emotion_dim: int = 256,
        action_dim: int = 256
    ):
        super().__init__()
        
        # Action encoder - processes proposed action
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Context integration
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # action + target_state + self_state
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Separate predictors for target and self
        self.target_emotion_predictor = EmotionDeltaPredictor(hidden_dim, emotion_dim)
        self.self_emotion_predictor = EmotionDeltaPredictor(hidden_dim, emotion_dim)
        
        # Valence and arousal prediction
        self.valence_arousal_predictor = ValenceArousalPredictor(hidden_dim)
        
        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        action_embedding: torch.Tensor,  # [batch, action_dim]
        target_state: torch.Tensor,  # [batch, hidden_dim]
        self_state: torch.Tensor,  # [batch, hidden_dim]
    ) -> EmotionalImpact:
        """
        Predict emotional impact of an action.
        
        Args:
            action_embedding: Embedding of proposed action
            target_state: Current state of target person
            self_state: Current internal state of AI
            
        Returns:
            EmotionalImpact with predicted emotional changes
        """
        # Encode action
        action_encoded = self.action_encoder(action_embedding)
        
        # Fuse contexts
        combined = torch.cat([action_encoded, target_state, self_state], dim=-1)
        fused_context = self.context_fusion(combined)
        
        # Predict emotion deltas
        target_emotion_delta = self.target_emotion_predictor(fused_context)
        self_emotion_delta = self.self_emotion_predictor(fused_context)
        
        # Predict valence and arousal
        valence, arousal = self.valence_arousal_predictor(fused_context)
        
        # Estimate confidence
        confidence = self.confidence_net(fused_context).squeeze(-1)
        
        return EmotionalImpact(
            target_emotion_delta=target_emotion_delta,
            self_emotion_delta=self_emotion_delta,
            valence=valence.item() if valence.dim() == 0 else valence.mean().item(),
            arousal=arousal.item() if arousal.dim() == 0 else arousal.mean().item(),
            confidence=confidence.item() if confidence.dim() == 0 else confidence.mean().item()
        )


class EmpathyModule(nn.Module):
    """Models empathic responses - feeling what others feel."""
    
    def __init__(self, emotion_dim: int = 256):
        super().__init__()
        
        # Mirror emotion network - simulates empathic mirroring
        self.mirror_net = nn.Sequential(
            nn.Linear(emotion_dim, emotion_dim),
            nn.ReLU(),
            nn.Linear(emotion_dim, emotion_dim),
            nn.Tanh(),
        )
        
        # Empathy strength modulator
        self.empathy_strength = nn.Parameter(torch.tensor(0.7))  # Learnable empathy level
        
    def forward(self, target_emotion: torch.Tensor) -> torch.Tensor:
        """
        Generate empathic emotional response.
        
        Args:
            target_emotion: Predicted emotion of target
            
        Returns:
            Mirrored emotion felt by AI
        """
        mirrored = self.mirror_net(target_emotion)
        # Scale by empathy strength
        return mirrored * torch.sigmoid(self.empathy_strength)


def create_emotion_module(config) -> EmotionImpactModule:
    """Factory function to create emotion impact module from config."""
    return EmotionImpactModule(
        hidden_dim=config.model_scale.hidden_dim,
        emotion_dim=config.model_scale.multimodal_dim // 4,
        action_dim=config.model_scale.action_dim
    )
