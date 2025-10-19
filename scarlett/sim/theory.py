"""Theory of mind simulation module.

Predicts mental states, beliefs, desires, and intentions of other agents.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import TheoryOfMindState


class BeliefPredictor(nn.Module):
    """Predicts beliefs and knowledge states of others."""
    
    def __init__(self, hidden_dim: int = 768, semantic_dim: int = 768):
        super().__init__()
        
        self.belief_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim),
        )
        
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Predict belief state from context."""
        return self.belief_net(context)


class DesirePredictor(nn.Module):
    """Predicts desires and goals of others."""
    
    def __init__(self, hidden_dim: int = 768, intent_dim: int = 512):
        super().__init__()
        
        self.desire_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, intent_dim),
        )
        
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Predict desire state from context."""
        return self.desire_net(context)


class EmotionPredictor(nn.Module):
    """Predicts emotional states of others."""
    
    def __init__(self, hidden_dim: int = 768, emotion_dim: int = 256):
        super().__init__()
        
        self.emotion_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, emotion_dim),
            nn.Tanh(),  # Normalize emotions
        )
        
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Predict emotional state from context."""
        return self.emotion_net(context)


class ActionPredictor(nn.Module):
    """Predicts likely actions of others."""
    
    def __init__(self, hidden_dim: int = 768, action_dim: int = 256):
        super().__init__()
        
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Predict action tendencies from context."""
        return self.action_net(context)


class TheoryOfMindModule(nn.Module):
    """Complete theory of mind simulation module."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        semantic_dim: int = 768,
        intent_dim: int = 512,
        emotion_dim: int = 256,
        action_dim: int = 256
    ):
        super().__init__()
        
        # Context encoder - processes multimodal input about target
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Individual predictors
        self.belief_predictor = BeliefPredictor(hidden_dim, semantic_dim)
        self.desire_predictor = DesirePredictor(hidden_dim, intent_dim)
        self.emotion_predictor = EmotionPredictor(hidden_dim, emotion_dim)
        self.action_predictor = ActionPredictor(hidden_dim, action_dim)
        
        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        perception_context: torch.Tensor,  # [batch, hidden_dim]
        historical_context: Optional[torch.Tensor] = None,  # [batch, hidden_dim]
    ) -> TheoryOfMindState:
        """
        Simulate theory of mind for target agent.
        
        Args:
            perception_context: Current perceptual information about target
            historical_context: Optional historical interaction context
            
        Returns:
            TheoryOfMindState with predicted mental state
        """
        # Combine current and historical context if available
        if historical_context is not None:
            context = (perception_context + historical_context) / 2
        else:
            context = perception_context
        
        # Encode context
        encoded_context = self.context_encoder(context)
        
        # Predict mental state components
        beliefs = self.belief_predictor(encoded_context)
        desires = self.desire_predictor(encoded_context)
        emotions = self.emotion_predictor(encoded_context)
        predicted_actions = self.action_predictor(encoded_context)
        
        # Estimate confidence
        confidence = self.confidence_net(encoded_context).squeeze(-1)
        
        return TheoryOfMindState(
            beliefs=beliefs,
            desires=desires,
            emotions=emotions,
            predicted_actions=predicted_actions,
            confidence=confidence.item() if confidence.dim() == 0 else confidence.mean().item()
        )


def create_theory_of_mind_module(config) -> TheoryOfMindModule:
    """Factory function to create theory of mind module from config."""
    return TheoryOfMindModule(
        hidden_dim=config.model_scale.hidden_dim,
        semantic_dim=config.model_scale.hidden_dim,  # Use hidden_dim for semantic
        intent_dim=config.model_scale.intent_dim,
        emotion_dim=config.model_scale.multimodal_dim // 4,  # Reasonable emotion dimension
        action_dim=config.model_scale.action_dim
    )
