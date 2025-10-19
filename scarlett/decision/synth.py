"""Decision synthesis module.

Integrates all evaluators (simulation, intent, moral) to produce final action logits.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import (
    ActionType, DecisionLogits, DecisionTrace, IntentResolved,
    SimulationResult, IntentDraft
)


class SimulationEvaluator(nn.Module):
    """Evaluates simulation results to produce evaluation scores."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Aggregate simulation components
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # theory, emotion, outcome, relation
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Evaluation score
        self.evaluator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Evaluation score [-1, 1]
        )
        
    def forward(self, simulation_result: SimulationResult) -> torch.Tensor:
        """Evaluate simulation result."""
        # Aggregate all simulation components
        components = []
        
        if simulation_result.theory_of_mind is not None:
            components.append(simulation_result.theory_of_mind.beliefs)
        else:
            components.append(torch.zeros(1, 768))  # Placeholder
        
        if simulation_result.emotional_impact is not None:
            components.append(simulation_result.emotional_impact.target_emotion_delta)
        else:
            components.append(torch.zeros_like(components[0]))
        
        if simulation_result.outcome_projection is not None:
            components.append(simulation_result.outcome_projection.consequence_embedding)
        else:
            components.append(torch.zeros_like(components[0]))
        
        if simulation_result.relationship_trajectory is not None:
            # Project to same dimension
            rel_delta = simulation_result.relationship_trajectory.relationship_delta
            if rel_delta.shape[-1] != components[0].shape[-1]:
                rel_delta = nn.Linear(
                    rel_delta.shape[-1],
                    components[0].shape[-1]
                ).to(rel_delta.device)(rel_delta)
            components.append(rel_delta)
        else:
            components.append(torch.zeros_like(components[0]))
        
        # Concatenate and aggregate
        combined = torch.cat(components, dim=-1)
        aggregated = self.aggregator(combined)
        
        # Evaluate
        return self.evaluator(aggregated).squeeze(-1)


class MoralIntegrator(nn.Module):
    """Integrates moral scores into decision making."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Moral bias network
        self.moral_bias = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),  # Moral score input
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, len(ActionType)),
            nn.Tanh(),  # Bias for each action type
        )
        
    def forward(self, moral_score: float) -> torch.Tensor:
        """Generate moral bias for action logits."""
        moral_tensor = torch.tensor([[moral_score]], dtype=torch.float32)
        return self.moral_bias(moral_tensor).squeeze(0)


class DecisionSynthesizer(nn.Module):
    """Synthesizes final decision from all inputs."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        action_dim: int = 256
    ):
        super().__init__()
        
        # Combine all decision factors
        self.decision_combiner = nn.Sequential(
            nn.Linear(action_dim + hidden_dim + 1, hidden_dim),  # action + simulation + moral
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Generate action logits
        self.action_logits_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, len(ActionType)),
        )
        
        # Moral tendency logits (kind vs mean)
        self.moral_logits_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2),  # Kind vs Mean
        )
        
        # Confidence logit
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        action_embedding: torch.Tensor,
        simulation_score: torch.Tensor,
        moral_score: torch.Tensor
    ) -> DecisionLogits:
        """
        Synthesize final decision logits.
        
        Args:
            action_embedding: Action representation from intent
            simulation_score: Aggregated simulation evaluation
            moral_score: Moral evaluation score
            
        Returns:
            DecisionLogits with action logits and metadata
        """
        # Ensure tensors have batch dimension
        if action_embedding.dim() == 1:
            action_embedding = action_embedding.unsqueeze(0)
        if simulation_score.dim() == 0:
            simulation_score = simulation_score.unsqueeze(0).unsqueeze(0)
        elif simulation_score.dim() == 1:
            simulation_score = simulation_score.unsqueeze(1)
        if moral_score.dim() == 0:
            moral_score = moral_score.unsqueeze(0).unsqueeze(0)
        elif moral_score.dim() == 1:
            moral_score = moral_score.unsqueeze(1)
        
        # Project simulation score to match dimensions if needed
        if simulation_score.shape[-1] != action_embedding.shape[-1]:
            sim_proj = nn.Linear(
                simulation_score.shape[-1],
                action_embedding.shape[-1]
            ).to(simulation_score.device)
            simulation_score = sim_proj(simulation_score)
        
        # Combine all factors
        combined = torch.cat([action_embedding, simulation_score, moral_score], dim=-1)
        decision_state = self.decision_combiner(combined)
        
        # Generate logits
        action_logits = self.action_logits_net(decision_state)
        moral_logits = self.moral_logits_net(decision_state)
        confidence = self.confidence_net(decision_state).squeeze(-1)
        
        return DecisionLogits(
            action_logits=action_logits.squeeze(0) if action_logits.shape[0] == 1 else action_logits,
            moral_logits=moral_logits.squeeze(0) if moral_logits.shape[0] == 1 else moral_logits,
            confidence_logit=confidence.item() if confidence.numel() == 1 else confidence.mean().item(),
            temperature=1.0
        )


class DecisionSynthesisModule(nn.Module):
    """Complete decision synthesis module."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        action_dim: int = 256
    ):
        super().__init__()
        
        self.simulation_evaluator = SimulationEvaluator(hidden_dim)
        self.moral_integrator = MoralIntegrator(hidden_dim)
        self.decision_synthesizer = DecisionSynthesizer(hidden_dim, action_dim)
        
    def forward(
        self,
        intent: IntentResolved,
        simulation: SimulationResult
    ) -> DecisionLogits:
        """
        Synthesize final decision.
        
        Args:
            intent: Resolved intent from intent formulation
            simulation: Simulation results
            
        Returns:
            DecisionLogits for final action selection
        """
        # Evaluate simulation
        sim_score = self.simulation_evaluator(simulation)
        
        # Get moral score from intent
        moral_score = torch.tensor([intent.moral_score], dtype=torch.float32)
        
        # Synthesize decision
        decision_logits = self.decision_synthesizer(
            intent.action_embedding,
            sim_score,
            moral_score
        )
        
        return decision_logits


def create_decision_module(config) -> DecisionSynthesisModule:
    """Factory function to create decision synthesis module from config."""
    return DecisionSynthesisModule(
        hidden_dim=config.model_scale.hidden_dim,
        action_dim=config.model_scale.action_dim
    )
