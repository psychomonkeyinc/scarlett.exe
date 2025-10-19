"""Goals management module.

Manages hierarchical goals (long-term, mid-term, immediate) and translates them into intent drafts.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import IntentDraft, ActionType


@dataclass
class Goal:
    """Represents a goal at any level."""
    embedding: torch.Tensor  # Goal representation
    priority: float  # 0.0 to 1.0
    time_horizon: str  # "immediate", "mid_term", "long_term"
    description: str = ""


class GoalEncoder(nn.Module):
    """Encodes goals from semantic descriptions."""
    
    def __init__(self, hidden_dim: int = 768, goal_dim: int = 512):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, goal_dim),
        )
        
    def forward(self, semantic_input: torch.Tensor) -> torch.Tensor:
        """Encode semantic input into goal embedding."""
        return self.encoder(semantic_input)


class GoalPrioritizer(nn.Module):
    """Prioritizes goals based on context and internal state."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Priority network
        self.priority_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Priority [0, 1]
        )
        
    def forward(
        self,
        goal_embedding: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """Compute goal priority."""
        combined = torch.cat([goal_embedding, context], dim=-1)
        return self.priority_net(combined).squeeze(-1)


class GoalDecomposer(nn.Module):
    """Decomposes long-term goals into mid-term and immediate goals."""
    
    def __init__(self, hidden_dim: int = 768, goal_dim: int = 512):
        super().__init__()
        
        # Decomposition network
        self.decompose_net = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim * 3),  # 3 sub-goals
        )
        
    def forward(self, goal_embedding: torch.Tensor) -> List[torch.Tensor]:
        """
        Decompose goal into sub-goals.
        
        Args:
            goal_embedding: High-level goal
            
        Returns:
            List of sub-goal embeddings
        """
        decomposed = self.decompose_net(goal_embedding)
        # Split into 3 sub-goals
        goal_dim = goal_embedding.shape[-1]
        return [
            decomposed[..., :goal_dim],
            decomposed[..., goal_dim:goal_dim*2],
            decomposed[..., goal_dim*2:],
        ]


class GoalToIntentTranslator(nn.Module):
    """Translates active goals into intent drafts."""
    
    def __init__(self, goal_dim: int = 512, intent_dim: int = 512):
        super().__init__()
        
        # Goal to intent transformation
        self.translator = nn.Sequential(
            nn.Linear(goal_dim, intent_dim),
            nn.LayerNorm(intent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intent_dim, intent_dim),
        )
        
        # Action type predictor
        self.action_predictor = nn.Sequential(
            nn.Linear(intent_dim, intent_dim // 2),
            nn.ReLU(),
            nn.Linear(intent_dim // 2, len(ActionType)),  # Predict action type
        )
        
    def forward(self, goal_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Translate goal to intent draft.
        
        Args:
            goal_embedding: Goal representation
            
        Returns:
            Dictionary with intent embedding and action logits
        """
        intent_embedding = self.translator(goal_embedding)
        action_logits = self.action_predictor(intent_embedding)
        
        return {
            'intent_embedding': intent_embedding,
            'action_logits': action_logits,
        }


class HierarchicalGoalManager(nn.Module):
    """Manages goals at multiple time horizons."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        goal_dim: int = 512,
        intent_dim: int = 512
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.goal_dim = goal_dim
        self.intent_dim = intent_dim
        
        # Components
        self.goal_encoder = GoalEncoder(hidden_dim, goal_dim)
        self.goal_prioritizer = GoalPrioritizer(hidden_dim)
        self.goal_decomposer = GoalDecomposer(hidden_dim, goal_dim)
        self.goal_to_intent = GoalToIntentTranslator(goal_dim, intent_dim)
        
        # Context-aware goal selection
        self.goal_selector = nn.Sequential(
            nn.Linear(hidden_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Selection probability
        )
        
    def forward(
        self,
        long_term_goals: Optional[List[torch.Tensor]] = None,
        mid_term_goals: Optional[List[torch.Tensor]] = None,
        immediate_goals: Optional[List[torch.Tensor]] = None,
        context: Optional[torch.Tensor] = None,
    ) -> List[IntentDraft]:
        """
        Manage goals and generate intent drafts.
        
        Args:
            long_term_goals: List of long-term goal embeddings
            mid_term_goals: List of mid-term goal embeddings
            immediate_goals: List of immediate goal embeddings
            context: Current context
            
        Returns:
            List of IntentDraft objects
        """
        intent_drafts = []
        
        # Process immediate goals (highest priority)
        if immediate_goals is not None and len(immediate_goals) > 0:
            for goal_emb in immediate_goals:
                if context is not None:
                    # Compute priority
                    priority = self.goal_prioritizer(goal_emb, context)
                else:
                    priority = torch.tensor(0.8)  # Default high priority for immediate
                
                # Translate to intent
                intent_result = self.goal_to_intent(goal_emb)
                
                # Determine action type (use argmax of logits)
                action_idx = torch.argmax(intent_result['action_logits'], dim=-1)
                action_type = list(ActionType)[action_idx.item() if action_idx.dim() == 0 else action_idx[0].item()]
                
                intent_drafts.append(IntentDraft(
                    goal_embedding=intent_result['intent_embedding'],
                    priority=priority.item() if priority.dim() == 0 else priority[0].item(),
                    action_type=action_type,
                    rationale="Immediate goal"
                ))
        
        # Process mid-term goals (medium priority)
        if mid_term_goals is not None and len(mid_term_goals) > 0:
            for goal_emb in mid_term_goals:
                if context is not None:
                    priority = self.goal_prioritizer(goal_emb, context) * 0.6  # Scale down
                else:
                    priority = torch.tensor(0.5)
                
                intent_result = self.goal_to_intent(goal_emb)
                action_idx = torch.argmax(intent_result['action_logits'], dim=-1)
                action_type = list(ActionType)[action_idx.item() if action_idx.dim() == 0 else action_idx[0].item()]
                
                intent_drafts.append(IntentDraft(
                    goal_embedding=intent_result['intent_embedding'],
                    priority=priority.item() if priority.dim() == 0 else priority[0].item(),
                    action_type=action_type,
                    rationale="Mid-term goal"
                ))
        
        # Process long-term goals (lower priority, may decompose)
        if long_term_goals is not None and len(long_term_goals) > 0:
            for goal_emb in long_term_goals:
                # Decompose into sub-goals
                sub_goals = self.goal_decomposer(goal_emb)
                
                # Process first sub-goal as immediate action
                if len(sub_goals) > 0:
                    if context is not None:
                        priority = self.goal_prioritizer(sub_goals[0], context) * 0.4
                    else:
                        priority = torch.tensor(0.3)
                    
                    intent_result = self.goal_to_intent(sub_goals[0])
                    action_idx = torch.argmax(intent_result['action_logits'], dim=-1)
                    action_type = list(ActionType)[action_idx.item() if action_idx.dim() == 0 else action_idx[0].item()]
                    
                    intent_drafts.append(IntentDraft(
                        goal_embedding=intent_result['intent_embedding'],
                        priority=priority.item() if priority.dim() == 0 else priority[0].item(),
                        action_type=action_type,
                        rationale="Long-term goal (decomposed)"
                    ))
        
        return intent_drafts


def create_goal_manager(config) -> HierarchicalGoalManager:
    """Factory function to create goal manager from config."""
    return HierarchicalGoalManager(
        hidden_dim=config.model_scale.hidden_dim,
        goal_dim=config.model_scale.intent_dim,
        intent_dim=config.model_scale.intent_dim
    )
