"""Intent formulation module.

Derives draft intents from embeddings and internal state, coordinating goals,
desires, and agency to produce coherent action intentions.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import IntentDraft, IntentResolved, ActionType


class IntentDraftGenerator(nn.Module):
    """Generates draft intentions from goals and context."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        intent_dim: int = 512
    ):
        super().__init__()
        
        # Combine multiple inputs
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # perception + goals + internal_state
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Generate intent embedding
        self.intent_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, intent_dim),
        )
        
        # Priority estimator
        self.priority_net = nn.Sequential(
            nn.Linear(intent_dim, intent_dim // 2),
            nn.ReLU(),
            nn.Linear(intent_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        perception: torch.Tensor,
        goals: torch.Tensor,
        internal_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generate intent draft."""
        combined = torch.cat([perception, goals, internal_state], dim=-1)
        fused = self.combiner(combined)
        
        intent_emb = self.intent_generator(fused)
        priority = self.priority_net(intent_emb).squeeze(-1)
        
        return {
            'intent_embedding': intent_emb,
            'priority': priority,
        }


class IntentRefiner(nn.Module):
    """Refines draft intentions using desire resolution and agency signals."""
    
    def __init__(
        self,
        intent_dim: int = 512,
        action_dim: int = 256
    ):
        super().__init__()
        
        # Refine intent with desires
        self.desire_integration = nn.Sequential(
            nn.Linear(intent_dim * 2, intent_dim),  # intent + desires
            nn.LayerNorm(intent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Agency modulation
        self.agency_modulator = nn.Sequential(
            nn.Linear(intent_dim + 1, intent_dim),  # intent + agency_strength
            nn.ReLU(),
        )
        
        # Project to action dimension
        self.action_projection = nn.Linear(intent_dim, action_dim)
        
        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(action_dim, action_dim // 2),
            nn.ReLU(),
            nn.Linear(action_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        intent_draft: torch.Tensor,
        desire_embedding: torch.Tensor,
        agency_strength: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Refine intent with desires and agency."""
        # Integrate desires
        combined = torch.cat([intent_draft, desire_embedding], dim=-1)
        desire_refined = self.desire_integration(combined)
        
        # Modulate by agency
        agency_input = torch.cat([
            desire_refined,
            agency_strength.unsqueeze(-1) if agency_strength.dim() == 0 or agency_strength.shape[-1] == 1 else agency_strength
        ], dim=-1)
        agency_modulated = self.agency_modulator(agency_input)
        
        # Project to action
        action_emb = self.action_projection(agency_modulated)
        
        # Estimate confidence
        confidence = self.confidence_net(action_emb).squeeze(-1)
        
        return {
            'action_embedding': action_emb,
            'confidence': confidence,
        }


class MoralEvaluator(nn.Module):
    """Evaluates moral implications of intentions."""
    
    def __init__(self, action_dim: int = 256):
        super().__init__()
        
        # Moral scoring network
        self.moral_net = nn.Sequential(
            nn.Linear(action_dim * 2, action_dim),  # action + context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(action_dim, action_dim // 2),
            nn.ReLU(),
            nn.Linear(action_dim // 2, 1),
            nn.Tanh(),  # Moral score [-1, 1]
        )
        
    def forward(
        self,
        action_embedding: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate moral score of action."""
        # Project context to action dimension if needed
        if context.shape[-1] != action_embedding.shape[-1]:
            context = nn.Linear(
                context.shape[-1],
                action_embedding.shape[-1]
            ).to(context.device)(context)
        
        combined = torch.cat([action_embedding, context], dim=-1)
        return self.moral_net(combined).squeeze(-1)


class IntentFormulationModule(nn.Module):
    """Complete intent formulation module."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        intent_dim: int = 512,
        action_dim: int = 256
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.intent_dim = intent_dim
        self.action_dim = action_dim
        
        # Components
        self.draft_generator = IntentDraftGenerator(hidden_dim, intent_dim)
        self.intent_refiner = IntentRefiner(intent_dim, action_dim)
        self.moral_evaluator = MoralEvaluator(action_dim)
        
        # Action type classifier
        self.action_classifier = nn.Sequential(
            nn.Linear(action_dim, action_dim // 2),
            nn.ReLU(),
            nn.Linear(action_dim // 2, len(ActionType)),
        )
        
    def forward(
        self,
        perception_embedding: torch.Tensor,
        goals_embedding: torch.Tensor,
        internal_state: torch.Tensor,
        desire_embedding: torch.Tensor,
        agency_strength: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> IntentResolved:
        """
        Formulate complete intention.
        
        Args:
            perception_embedding: Perceptual input
            goals_embedding: Active goals
            internal_state: Internal state
            desire_embedding: Resolved desires
            agency_strength: Agency activation level
            context: Optional additional context
            
        Returns:
            IntentResolved with final intention
        """
        # Generate draft intent
        draft = self.draft_generator(
            perception_embedding,
            goals_embedding,
            internal_state
        )
        
        # Refine with desires and agency
        refined = self.intent_refiner(
            draft['intent_embedding'],
            desire_embedding,
            agency_strength
        )
        
        # Classify action type
        action_logits = self.action_classifier(refined['action_embedding'])
        action_idx = torch.argmax(action_logits, dim=-1)
        action_type = list(ActionType)[action_idx.item() if action_idx.dim() == 0 else action_idx[0].item()]
        
        # Evaluate moral implications
        if context is None:
            context = perception_embedding
        moral_score = self.moral_evaluator(refined['action_embedding'], context)
        
        return IntentResolved(
            action_type=action_type,
            action_embedding=refined['action_embedding'],
            confidence=refined['confidence'].item() if refined['confidence'].dim() == 0 else refined['confidence'].mean().item(),
            moral_score=moral_score.item() if moral_score.dim() == 0 else moral_score.mean().item(),
            rationale=f"Intent formulated from goals with {action_type.value} action"
        )


def create_intent_module(config) -> IntentFormulationModule:
    """Factory function to create intent formulation module from config."""
    return IntentFormulationModule(
        hidden_dim=config.model_scale.hidden_dim,
        intent_dim=config.model_scale.intent_dim,
        action_dim=config.model_scale.action_dim
    )
