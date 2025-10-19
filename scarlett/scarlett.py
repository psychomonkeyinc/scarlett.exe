"""Scarlett AI - Complete Integration Module.

Integrates all components into a complete decision-making system.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import time

from core.config import ScarlettConfig
from core.types import (
    ActionType, ContextualInput, InternalState, IntentDraft, IntentResolved,
    SimulationResult, DecisionTrace, DecisionLogits
)

# Import all modules
from perception.audio import create_audio_module
from perception.vision import create_vision_module
from perception.fuse import create_fusion_module
from sim.theory import create_theory_of_mind_module
from sim.emotion import create_emotion_module
from sim.outcome import create_outcome_module
from sim.relation import create_relation_module
from intent.desire import create_desire_engine
from intent.agency import create_agency_module
from intent.goals import create_goal_manager
from intent.intent import create_intent_module
from decision.actions import create_action_space
from decision.synth import create_decision_module
from safety.guardrails import create_safety_system


class ScarlettAI(nn.Module):
    """
    Complete Scarlett AI system.
    
    Integrates perception, simulation, intent formulation, decision synthesis,
    and safety guardrails into a unified decision-making architecture.
    """
    
    def __init__(self, config: ScarlettConfig):
        super().__init__()
        
        self.config = config
        
        # Set device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Initialize all modules
        print(f"Initializing Scarlett AI ({config.model_scale.name})...")
        
        # Perception modules
        self.audio_module = create_audio_module(config).to(self.device)
        self.vision_module = create_vision_module(config).to(self.device)
        self.fusion_module = create_fusion_module(config).to(self.device)
        
        # Simulation modules
        self.theory_of_mind = create_theory_of_mind_module(config).to(self.device)
        self.emotion_module = create_emotion_module(config).to(self.device)
        self.outcome_module = create_outcome_module(config).to(self.device)
        self.relation_module = create_relation_module(config).to(self.device)
        
        # Intent formulation modules
        self.desire_engine = create_desire_engine(config).to(self.device)
        self.agency_module = create_agency_module(config).to(self.device)
        self.goal_manager = create_goal_manager(config).to(self.device)
        self.intent_module = create_intent_module(config).to(self.device)
        
        # Decision modules
        self.action_space = create_action_space(config).to(self.device)
        self.decision_module = create_decision_module(config).to(self.device)
        
        # Safety system
        self.safety_system = create_safety_system(config).to(self.device)
        
        # Decision counter for self-testing
        self.decision_count = 0
        
        print(f"Scarlett AI initialized successfully on {self.device}")
        print(f"Total parameters: ~{config.model_scale.total_params}")
    
    def perceive(
        self,
        audio_features: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Process multimodal perceptual input.
        
        Args:
            audio_features: Audio features (MFCC, etc.)
            image_features: Image features (from ViT, etc.)
            **kwargs: Additional perception parameters
            
        Returns:
            Dictionary with fused perception results
        """
        # Process audio
        audio_output = None
        if audio_features is not None:
            audio_output = self.audio_module(
                audio_features,
                return_details=kwargs.get('return_audio_details', False)
            )
        
        # Process vision
        vision_output = None
        if image_features is not None:
            vision_output = self.vision_module(
                image_features,
                return_details=kwargs.get('return_vision_details', False)
            )
        
        # Fuse modalities
        if audio_output is not None and vision_output is not None:
            fusion_result = self.fusion_module(
                audio_output,
                vision_output,
                return_details=kwargs.get('return_fusion_details', False)
            )
            return fusion_result
        elif audio_output is not None:
            return {'perception_embedding': audio_output['embedding']}
        elif vision_output is not None:
            return {'perception_embedding': vision_output['embedding']}
        else:
            # No input - return zero embedding
            return {
                'perception_embedding': torch.zeros(
                    1, self.config.model_scale.multimodal_dim,
                    device=self.device
                )
            }
    
    def simulate(
        self,
        action_embedding: torch.Tensor,
        perception_context: torch.Tensor,
        internal_state: InternalState
    ) -> SimulationResult:
        """
        Simulate consequences of an action.
        
        Args:
            action_embedding: Proposed action
            perception_context: Current perceptual context
            internal_state: Current internal state
            
        Returns:
            SimulationResult with all predictions
        """
        # Theory of mind - predict target's mental state
        theory_of_mind = self.theory_of_mind(
            perception_context,
            historical_context=None  # Could add memory here
        )
        
        # Emotional impact
        target_state = perception_context  # Simplified
        self_state = internal_state.emotional_state.unsqueeze(0) if internal_state.emotional_state.dim() == 1 else internal_state.emotional_state
        emotional_impact = self.emotion_module(
            action_embedding,
            target_state,
            self_state
        )
        
        # Outcome projection
        outcome_projection = self.outcome_module(
            action_embedding,
            perception_context,
            historical_context=None
        )
        
        # Relationship trajectory
        # Get current relationship from internal state
        current_relationship = list(internal_state.relationship_memory.values())[0] if internal_state.relationship_memory else torch.zeros(
            1, self.config.model_scale.memory_dim, device=self.device
        )
        relationship_trajectory = self.relation_module(
            action_embedding,
            current_relationship,
            interaction_history=None
        )
        
        # Aggregate simulation results
        overall_score = (
            0.3 * (emotional_impact.valence + 1.0) / 2.0 +  # Normalize to [0,1]
            0.3 * outcome_projection.success_probability +
            0.2 * (relationship_trajectory.bond_strength) +
            0.2 * (1.0 - emotional_impact.arousal)  # Prefer calm outcomes
        )
        
        confidence = (
            theory_of_mind.confidence +
            emotional_impact.confidence +
            outcome_projection.confidence +
            relationship_trajectory.confidence
        ) / 4.0
        
        return SimulationResult(
            theory_of_mind=theory_of_mind,
            emotional_impact=emotional_impact,
            outcome_projection=outcome_projection,
            relationship_trajectory=relationship_trajectory,
            overall_score=overall_score,
            confidence=confidence
        )
    
    def formulate_intent(
        self,
        perception_embedding: torch.Tensor,
        internal_state: InternalState,
        goals: Optional[List[torch.Tensor]] = None
    ) -> IntentResolved:
        """
        Formulate intention from perception and internal state.
        
        Args:
            perception_embedding: Fused perceptual input
            internal_state: Current internal state
            goals: Optional list of goal embeddings
            
        Returns:
            IntentResolved with action intention
        """
        # Project perception to hidden_dim if needed
        if perception_embedding.shape[-1] != self.config.model_scale.hidden_dim:
            proj = nn.Linear(
                perception_embedding.shape[-1],
                self.config.model_scale.hidden_dim
            ).to(self.device)
            perception_proj = proj(perception_embedding)
        else:
            perception_proj = perception_embedding
        
        # Resolve desires
        desire_result = self.desire_engine(
            perception_proj,
            internal_state.emotional_state.unsqueeze(0) if internal_state.emotional_state.dim() == 1 else internal_state.emotional_state
        )
        
        # Compute agency
        agency_result = self.agency_module(
            internal_state.emotional_state.unsqueeze(0) if internal_state.emotional_state.dim() == 1 else internal_state.emotional_state,
            perception_proj
        )
        
        # Process goals (if provided, else use perception as goal)
        if goals is not None:
            intent_drafts = self.goal_manager(
                immediate_goals=goals,
                context=perception_proj
            )
            # Use first goal's embedding
            goals_embedding = intent_drafts[0].goal_embedding if intent_drafts else perception_proj
        else:
            goals_embedding = perception_proj
        
        # Formulate intent
        intent = self.intent_module(
            perception_proj,
            goals_embedding,
            internal_state.emotional_state.unsqueeze(0) if internal_state.emotional_state.dim() == 1 else internal_state.emotional_state,
            desire_result['desire_embedding'],
            agency_result['activation_readiness'],
            context=perception_proj
        )
        
        return intent
    
    def decide(
        self,
        contextual_input: ContextualInput,
        internal_state: InternalState,
        goals: Optional[List[torch.Tensor]] = None,
        return_trace: bool = False
    ) -> tuple:
        """
        Make a decision based on context and internal state.
        
        Args:
            contextual_input: Current context
            internal_state: Current internal state
            goals: Optional goals
            return_trace: Whether to return decision trace
            
        Returns:
            Tuple of (action_type, decision_trace if requested)
        """
        start_time = time.time()
        
        # Perceive (use situation encoding as perception)
        perception_result = {
            'perception_embedding': contextual_input.situation_encoding.unsqueeze(0) if contextual_input.situation_encoding.dim() == 1 else contextual_input.situation_encoding
        }
        perception_embedding = perception_result['perception_embedding']
        
        # Formulate intent
        intent = self.formulate_intent(
            perception_embedding,
            internal_state,
            goals
        )
        
        # Simulate consequences
        simulation = self.simulate(
            intent.action_embedding.unsqueeze(0) if intent.action_embedding.dim() == 1 else intent.action_embedding,
            perception_embedding,
            internal_state
        )
        
        # Store simulation in intent
        intent.expected_outcome = simulation
        
        # Synthesize decision
        decision_logits = self.decision_module(intent, simulation)
        
        # Safety check
        is_safe, violations, safety_scores = self.safety_system.check_safety(
            intent,
            perception_embedding,
            historical_actions=None  # Could add action history
        )
        
        # Apply safety corrections if needed
        if violations and self.config.safety.enable_guardrails:
            decision_logits = self.safety_system.apply_correction(
                decision_logits,
                violations
            )
        
        # Select final action
        action_probs = torch.softmax(decision_logits.action_logits / decision_logits.temperature, dim=-1)
        action_idx = torch.argmax(action_probs)
        final_action = list(ActionType)[action_idx.item()]
        
        # Increment decision counter
        self.decision_count += 1
        
        # Check if self-test needed
        needs_self_test = (
            self.config.safety.enable_self_testing and
            self.decision_count % self.config.safety.self_test_frequency == 0
        )
        
        if needs_self_test:
            final_action = ActionType.SELF_TEST
        
        # Create decision trace
        processing_time = time.time() - start_time
        
        trace = DecisionTrace(
            intent_draft=IntentDraft(
                goal_embedding=perception_embedding.squeeze(0) if perception_embedding.shape[0] == 1 else perception_embedding[0],
                priority=0.5,
                action_type=intent.action_type,
                rationale="Generated from perception"
            ),
            intent_resolved=intent,
            simulation=simulation,
            final_action=final_action,
            decision_logits=decision_logits,
            expert_activations={
                'perception': 1.0,
                'intent': 1.0,
                'simulation': 1.0,
                'decision': 1.0,
                'safety': 1.0 if violations else 0.0,
            },
            processing_time=processing_time,
            metadata={
                'is_safe': is_safe,
                'violations': [v.value for v in violations],
                'safety_scores': safety_scores,
                'needs_self_test': needs_self_test,
            }
        )
        
        if return_trace:
            return final_action, trace
        else:
            return final_action, None
    
    def forward(
        self,
        contextual_input: ContextualInput,
        internal_state: InternalState,
        **kwargs
    ) -> ActionType:
        """
        Forward pass - make a decision.
        
        Args:
            contextual_input: Current context
            internal_state: Current internal state
            **kwargs: Additional parameters
            
        Returns:
            Selected action type
        """
        action, _ = self.decide(contextual_input, internal_state, **kwargs)
        return action


def create_scarlett_ai(config: Optional[ScarlettConfig] = None) -> ScarlettAI:
    """
    Factory function to create Scarlett AI system.
    
    Args:
        config: Optional configuration (defaults to 500M scale)
        
    Returns:
        Initialized Scarlett AI system
    """
    if config is None:
        from core.config import get_default_config
        config = get_default_config("500M")
    
    return ScarlettAI(config)
