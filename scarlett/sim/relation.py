"""Relationship trajectory projection module.

Predicts how relationships evolve over time based on actions.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import RelationshipTrajectory


class RelationshipDeltaPredictor(nn.Module):
    """Predicts changes in relationship state."""
    
    def __init__(self, hidden_dim: int = 768, memory_dim: int = 512):
        super().__init__()
        
        self.delta_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, memory_dim),
            nn.Tanh(),  # Deltas in [-1, 1]
        )
        
    def forward(self, action_context: torch.Tensor) -> torch.Tensor:
        """Predict relationship delta."""
        return self.delta_net(action_context)


class TrustDeltaPredictor(nn.Module):
    """Predicts changes in trust level."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.trust_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Trust delta in [-1, 1]
        )
        
    def forward(self, action_context: torch.Tensor) -> torch.Tensor:
        """Predict trust delta."""
        return self.trust_net(action_context).squeeze(-1)


class AffectionDeltaPredictor(nn.Module):
    """Predicts changes in affection/closeness."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.affection_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Affection delta in [-1, 1]
        )
        
    def forward(self, action_context: torch.Tensor) -> torch.Tensor:
        """Predict affection delta."""
        return self.affection_net(action_context).squeeze(-1)


class BondStrengthPredictor(nn.Module):
    """Predicts overall bond strength after action."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.bond_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Bond strength in [0, 1]
        )
        
    def forward(self, action_context: torch.Tensor) -> torch.Tensor:
        """Predict bond strength."""
        return self.bond_net(action_context).squeeze(-1)


class RelationshipTrajectoryModule(nn.Module):
    """Complete relationship trajectory prediction module."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        memory_dim: int = 512,
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
            nn.Linear(hidden_dim * 3, hidden_dim),  # action + current_relationship + history
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Relationship predictors
        self.relationship_delta_predictor = RelationshipDeltaPredictor(hidden_dim, memory_dim)
        self.trust_delta_predictor = TrustDeltaPredictor(hidden_dim)
        self.affection_delta_predictor = AffectionDeltaPredictor(hidden_dim)
        self.bond_strength_predictor = BondStrengthPredictor(hidden_dim)
        
        # Time horizon predictor
        self.time_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )
        
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
        current_relationship: torch.Tensor,  # [batch, memory_dim]
        interaction_history: Optional[torch.Tensor] = None,  # [batch, hidden_dim]
    ) -> RelationshipTrajectory:
        """
        Project relationship trajectory.
        
        Args:
            action_embedding: Embedding of proposed action
            current_relationship: Current relationship state
            interaction_history: Optional interaction history
            
        Returns:
            RelationshipTrajectory with predicted changes
        """
        # Encode action
        action_encoded = self.action_encoder(action_embedding)
        
        # Project current relationship to hidden dim if needed
        if current_relationship.shape[-1] != action_encoded.shape[-1]:
            # Simple projection
            relationship_proj = nn.Linear(
                current_relationship.shape[-1],
                action_encoded.shape[-1]
            ).to(current_relationship.device)(current_relationship)
        else:
            relationship_proj = current_relationship
        
        # Handle optional history
        if interaction_history is None:
            interaction_history = torch.zeros_like(action_encoded)
        
        # Fuse contexts
        combined = torch.cat([action_encoded, relationship_proj, interaction_history], dim=-1)
        fused_context = self.context_fusion(combined)
        
        # Predict relationship changes
        relationship_delta = self.relationship_delta_predictor(fused_context)
        trust_delta = self.trust_delta_predictor(fused_context)
        affection_delta = self.affection_delta_predictor(fused_context)
        bond_strength = self.bond_strength_predictor(fused_context)
        time_horizon = self.time_net(fused_context).squeeze(-1)
        
        # Estimate confidence
        confidence = self.confidence_net(fused_context).squeeze(-1)
        
        return RelationshipTrajectory(
            relationship_delta=relationship_delta,
            trust_delta=trust_delta.item() if trust_delta.dim() == 0 else trust_delta.mean().item(),
            affection_delta=affection_delta.item() if affection_delta.dim() == 0 else affection_delta.mean().item(),
            bond_strength=bond_strength.item() if bond_strength.dim() == 0 else bond_strength.mean().item(),
            time_horizon=time_horizon.item() if time_horizon.dim() == 0 else time_horizon.mean().item(),
            confidence=confidence.item() if confidence.dim() == 0 else confidence.mean().item()
        )


class AttachmentStyleModel(nn.Module):
    """Models attachment styles (secure, anxious, avoidant) in relationships."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Attachment style classifier
        self.style_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3 attachment styles
            nn.Softmax(dim=-1),
        )
        
        # Attachment-specific response modulator
        self.response_modulator = nn.Linear(3, 1)
        
    def forward(
        self,
        relationship_state: torch.Tensor
    ) -> tuple:
        """
        Predict attachment style and modulation factor.
        
        Args:
            relationship_state: Current relationship state
            
        Returns:
            Tuple of (style_probs, modulation_factor)
        """
        style_probs = self.style_net(relationship_state)
        modulation = self.response_modulator(style_probs)
        return style_probs, modulation


class LongTermBondPredictor(nn.Module):
    """Predicts long-term relationship evolution."""
    
    def __init__(self, memory_dim: int = 512, num_steps: int = 10):
        super().__init__()
        self.num_steps = num_steps
        
        # Recurrent bond evolution
        self.evolution_rnn = nn.GRU(memory_dim, memory_dim, batch_first=True)
        
    def forward(
        self,
        initial_relationship: torch.Tensor,
        relationship_delta: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Predict long-term relationship evolution.
        
        Args:
            initial_relationship: Starting relationship state
            relationship_delta: Predicted immediate change
            num_steps: Number of steps to project (default: self.num_steps)
            
        Returns:
            Final predicted relationship state
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        # Start with modified state
        current = initial_relationship + relationship_delta
        
        # Project forward multiple steps
        # Create sequence by repeating (simplified)
        sequence = current.unsqueeze(1).repeat(1, num_steps, 1)
        
        # Evolve through RNN
        output, final_state = self.evolution_rnn(sequence)
        
        return final_state.squeeze(0)


def create_relation_module(config) -> RelationshipTrajectoryModule:
    """Factory function to create relationship trajectory module from config."""
    return RelationshipTrajectoryModule(
        hidden_dim=config.model_scale.hidden_dim,
        memory_dim=config.model_scale.memory_dim,
        action_dim=config.model_scale.action_dim
    )
