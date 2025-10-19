"""Desire conflict resolution engine.

Resolves competing internal desires and objectives using multi-objective optimization.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DesireEncoder(nn.Module):
    """Encodes different types of desires."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Different desire types
        self.consistency_encoder = nn.Linear(hidden_dim, hidden_dim // 4)
        self.novelty_encoder = nn.Linear(hidden_dim, hidden_dim // 4)
        self.risk_encoder = nn.Linear(hidden_dim, hidden_dim // 4)
        self.kindness_encoder = nn.Linear(hidden_dim, hidden_dim // 4)
        
    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode different desire types from context."""
        return {
            'consistency': self.consistency_encoder(context),
            'novelty': self.novelty_encoder(context),
            'risk': self.risk_encoder(context),
            'kindness': self.kindness_encoder(context),
        }


class DesireWeightingNetwork(nn.Module):
    """Learns to weight competing desires based on context."""
    
    def __init__(self, hidden_dim: int = 768, num_desires: int = 4):
        super().__init__()
        self.num_desires = num_desires
        
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_desires),
            nn.Softmax(dim=-1),  # Weights sum to 1
        )
        
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compute desire weights."""
        return self.weight_net(context)


class ConflictResolver(nn.Module):
    """Resolves conflicts between desires using learned strategies."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Conflict detection
        self.conflict_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Conflict intensity [0, 1]
        )
        
        # Resolution strategy network
        self.resolution_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(
        self,
        desire_embeddings: List[torch.Tensor]
    ) -> tuple:
        """
        Detect and resolve conflicts.
        
        Args:
            desire_embeddings: List of desire embeddings
            
        Returns:
            Tuple of (conflict_intensity, resolved_embedding)
        """
        # Combine all desires
        combined = torch.cat(desire_embeddings, dim=-1) if len(desire_embeddings) > 1 else desire_embeddings[0]
        
        # Detect conflict
        conflict = self.conflict_detector(combined)
        
        # Resolve through network
        resolved = self.resolution_net(combined)
        
        return conflict, resolved


class DesireConflictEngine(nn.Module):
    """Complete desire conflict resolution engine."""
    
    def __init__(self, hidden_dim: int = 768, num_desires: int = 4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_desires = num_desires
        
        # Components
        self.desire_encoder = DesireEncoder(hidden_dim)
        self.desire_weighting = DesireWeightingNetwork(hidden_dim, num_desires)
        self.conflict_resolver = ConflictResolver(hidden_dim)
        
        # Final desire representation
        self.desire_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        context: torch.Tensor,  # [batch, hidden_dim]
        internal_state: Optional[torch.Tensor] = None,  # [batch, emotion_dim or hidden_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Process and resolve desire conflicts.
        
        Args:
            context: Current context
            internal_state: Optional internal state
            
        Returns:
            Dictionary with resolved desires and metadata
        """
        # Encode different desire types
        desire_dict = self.desire_encoder(context)
        
        # Compute desire weights based on context and internal state
        if internal_state is not None:
            # Project internal state to match context dimension if needed
            if internal_state.shape[-1] != context.shape[-1]:
                proj = nn.Linear(internal_state.shape[-1], context.shape[-1]).to(context.device)
                internal_proj = proj(internal_state)
            else:
                internal_proj = internal_state
            combined_context = (context + internal_proj) / 2
        else:
            combined_context = context
        
        desire_weights = self.desire_weighting(combined_context)
        
        # Create weighted desire embeddings
        desire_embeddings = []
        desire_names = ['consistency', 'novelty', 'risk', 'kindness']
        for i, name in enumerate(desire_names):
            weighted = desire_dict[name] * desire_weights[:, i:i+1]
            desire_embeddings.append(weighted)
        
        # Detect and resolve conflicts
        conflict_intensity, resolved_embedding = self.conflict_resolver(desire_embeddings)
        
        # Final desire representation
        final_desire = self.desire_projection(resolved_embedding)
        
        return {
            'desire_embedding': final_desire,
            'desire_weights': desire_weights,
            'conflict_intensity': conflict_intensity,
            'individual_desires': desire_dict,
        }


class UtilityFunction(nn.Module):
    """Learns utility functions for different objectives."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Separate utility networks for different objectives
        self.kindness_utility = self._make_utility_net(hidden_dim)
        self.consistency_utility = self._make_utility_net(hidden_dim)
        self.novelty_utility = self._make_utility_net(hidden_dim)
        self.safety_utility = self._make_utility_net(hidden_dim)
        
    def _make_utility_net(self, hidden_dim: int) -> nn.Module:
        """Create a utility network."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Utility in [-1, 1]
        )
    
    def forward(
        self,
        action_embedding: torch.Tensor,
        context: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute utility for different objectives.
        
        Args:
            action_embedding: Proposed action
            context: Current context
            
        Returns:
            Dictionary of utility scores
        """
        # Combine action and context
        combined = (action_embedding + context) / 2
        
        return {
            'kindness': self.kindness_utility(combined).squeeze(-1),
            'consistency': self.consistency_utility(combined).squeeze(-1),
            'novelty': self.novelty_utility(combined).squeeze(-1),
            'safety': self.safety_utility(combined).squeeze(-1),
        }


def create_desire_engine(config) -> DesireConflictEngine:
    """Factory function to create desire conflict engine from config."""
    return DesireConflictEngine(
        hidden_dim=config.model_scale.hidden_dim,
        num_desires=4  # consistency, novelty, risk, kindness
    )
