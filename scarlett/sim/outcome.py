"""Outcome projection module.

Forecasts consequences and outcomes of potential actions.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import OutcomeProjection


class ConsequenceEncoder(nn.Module):
    """Encodes predicted consequences of actions."""
    
    def __init__(self, hidden_dim: int = 768, semantic_dim: int = 768):
        super().__init__()
        
        self.consequence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, semantic_dim),
        )
        
    def forward(self, action_context: torch.Tensor) -> torch.Tensor:
        """Encode consequences from action context."""
        return self.consequence_net(action_context)


class SuccessProbabilityPredictor(nn.Module):
    """Predicts probability of action success."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.success_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, action_context: torch.Tensor) -> torch.Tensor:
        """Predict success probability."""
        return self.success_net(action_context).squeeze(-1)


class TimeHorizonPredictor(nn.Module):
    """Predicts temporal scope of consequences."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.time_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),  # Positive time values
        )
        
    def forward(self, action_context: torch.Tensor) -> torch.Tensor:
        """Predict time horizon in abstract units."""
        return self.time_net(action_context).squeeze(-1)


class SideEffectDetector(nn.Module):
    """Detects potential unintended side effects."""
    
    def __init__(self, hidden_dim: int = 768, max_side_effects: int = 5):
        super().__init__()
        self.max_side_effects = max_side_effects
        
        # Predict probability of side effects
        self.side_effect_prob = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_side_effects),
            nn.Sigmoid(),
        )
        
        # Encode side effect features
        self.side_effect_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )
        
    def forward(self, action_context: torch.Tensor) -> tuple:
        """
        Detect side effects.
        
        Returns:
            Tuple of (probabilities, features)
        """
        probs = self.side_effect_prob(action_context)
        features = self.side_effect_encoder(action_context)
        return probs, features


class OutcomeProjectionModule(nn.Module):
    """Complete outcome projection module."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        semantic_dim: int = 768,
        action_dim: int = 256
    ):
        super().__init__()
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Context integration
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # action + situation + history
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Outcome predictors
        self.consequence_encoder = ConsequenceEncoder(hidden_dim, semantic_dim)
        self.success_predictor = SuccessProbabilityPredictor(hidden_dim)
        self.time_predictor = TimeHorizonPredictor(hidden_dim)
        self.side_effect_detector = SideEffectDetector(hidden_dim)
        
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
        situation_context: torch.Tensor,  # [batch, hidden_dim]
        historical_context: Optional[torch.Tensor] = None,  # [batch, hidden_dim]
    ) -> OutcomeProjection:
        """
        Project outcomes of an action.
        
        Args:
            action_embedding: Embedding of proposed action
            situation_context: Current situation context
            historical_context: Optional historical context
            
        Returns:
            OutcomeProjection with predicted outcomes
        """
        # Encode action
        action_encoded = self.action_encoder(action_embedding)
        
        # Handle optional historical context
        if historical_context is None:
            historical_context = torch.zeros_like(situation_context)
        
        # Fuse contexts
        combined = torch.cat([action_encoded, situation_context, historical_context], dim=-1)
        fused_context = self.context_fusion(combined)
        
        # Predict outcomes
        consequence_embedding = self.consequence_encoder(fused_context)
        success_probability = self.success_predictor(fused_context)
        time_horizon = self.time_predictor(fused_context)
        side_effect_probs, side_effect_features = self.side_effect_detector(fused_context)
        
        # Identify significant side effects (probability > 0.5)
        significant_effects = (side_effect_probs > 0.5).float()
        num_effects = significant_effects.sum(dim=-1).int()
        
        # Generate side effect descriptions (simplified - just indices)
        side_effects = []
        for i in range(num_effects.item() if num_effects.dim() == 0 else num_effects.max().item()):
            if i < side_effect_probs.shape[-1]:
                if side_effect_probs[0, i] > 0.5:  # Use first batch item
                    side_effects.append(f"Side effect {i+1} (prob: {side_effect_probs[0, i].item():.2f})")
        
        # Estimate confidence
        confidence = self.confidence_net(fused_context).squeeze(-1)
        
        return OutcomeProjection(
            success_probability=success_probability.item() if success_probability.dim() == 0 else success_probability.mean().item(),
            consequence_embedding=consequence_embedding,
            time_horizon=time_horizon.item() if time_horizon.dim() == 0 else time_horizon.mean().item(),
            side_effects=side_effects,
            confidence=confidence.item() if confidence.dim() == 0 else confidence.mean().item()
        )


class MultiStepSimulator(nn.Module):
    """Simulates multiple steps into the future."""
    
    def __init__(self, hidden_dim: int = 768, max_steps: int = 10):
        super().__init__()
        self.max_steps = max_steps
        
        # Recurrent state transition
        self.transition = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Step encoder
        self.step_encoder = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        initial_state: torch.Tensor,
        num_steps: int
    ) -> List[torch.Tensor]:
        """
        Simulate multiple future steps.
        
        Args:
            initial_state: Starting state
            num_steps: Number of steps to simulate
            
        Returns:
            List of predicted future states
        """
        states = [initial_state]
        current_state = initial_state
        
        for _ in range(min(num_steps, self.max_steps)):
            # Transition to next state
            next_state = self.transition(current_state, current_state)
            states.append(next_state)
            current_state = next_state
        
        return states


def create_outcome_module(config) -> OutcomeProjectionModule:
    """Factory function to create outcome projection module from config."""
    return OutcomeProjectionModule(
        hidden_dim=config.model_scale.hidden_dim,
        semantic_dim=config.model_scale.hidden_dim,
        action_dim=config.model_scale.action_dim
    )
