"""Action space definition.

Defines the concrete action types and their representations.
"""
import torch
import torch.nn as nn
from typing import Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import ActionType


class ActionEmbedding(nn.Module):
    """Learns embeddings for each action type."""
    
    def __init__(self, action_dim: int = 256):
        super().__init__()
        
        # Learnable embeddings for each action type
        self.action_embeddings = nn.Parameter(
            torch.randn(len(ActionType), action_dim)
        )
        
    def forward(self, action_type: ActionType) -> torch.Tensor:
        """Get embedding for action type."""
        action_idx = list(ActionType).index(action_type)
        return self.action_embeddings[action_idx]
    
    def get_all_embeddings(self) -> torch.Tensor:
        """Get all action embeddings."""
        return self.action_embeddings


class ActionParameterizer(nn.Module):
    """Generates parameters for actions based on context."""
    
    def __init__(
        self,
        action_dim: int = 256,
        param_dim: int = 128
    ):
        super().__init__()
        
        # Different action types may need different parameters
        self.kind_params = nn.Sequential(
            nn.Linear(action_dim, param_dim),
            nn.ReLU(),
            nn.Linear(param_dim, param_dim),
        )
        
        self.mean_params = nn.Sequential(
            nn.Linear(action_dim, param_dim),
            nn.ReLU(),
            nn.Linear(param_dim, param_dim),
        )
        
        self.neutral_params = nn.Sequential(
            nn.Linear(action_dim, param_dim),
            nn.ReLU(),
            nn.Linear(param_dim, param_dim),
        )
        
    def forward(
        self,
        action_embedding: torch.Tensor,
        action_type: ActionType
    ) -> torch.Tensor:
        """Generate parameters for specific action type."""
        if action_type == ActionType.KIND:
            return self.kind_params(action_embedding)
        elif action_type == ActionType.MEAN:
            return self.mean_params(action_embedding)
        else:
            return self.neutral_params(action_embedding)


class ActionSpace(nn.Module):
    """Complete action space module."""
    
    def __init__(
        self,
        action_dim: int = 256,
        param_dim: int = 128
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.param_dim = param_dim
        
        self.action_embedding = ActionEmbedding(action_dim)
        self.action_parameterizer = ActionParameterizer(action_dim, param_dim)
        
    def get_action_embedding(self, action_type: ActionType) -> torch.Tensor:
        """Get embedding for action type."""
        return self.action_embedding(action_type)
    
    def get_all_action_embeddings(self) -> torch.Tensor:
        """Get all action embeddings."""
        return self.action_embedding.get_all_embeddings()
    
    def parameterize_action(
        self,
        action_embedding: torch.Tensor,
        action_type: ActionType
    ) -> torch.Tensor:
        """Generate parameters for action."""
        return self.action_parameterizer(action_embedding, action_type)


def create_action_space(config) -> ActionSpace:
    """Factory function to create action space from config."""
    return ActionSpace(
        action_dim=config.model_scale.action_dim,
        param_dim=config.model_scale.action_dim // 2
    )
