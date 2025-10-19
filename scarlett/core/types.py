"""Core shared types for Scarlett AI architecture.

Defines fundamental data structures used throughout the system.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import torch


class ActionType(Enum):
    """Possible action types the AI can take."""
    KIND = "kind"
    MEAN = "mean"
    NEUTRAL = "neutral"
    SELF_TEST = "self_test"
    OBSERVE = "observe"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding dimensions across the system."""
    perception_dim: int = 1024  # Multimodal perception embedding
    semantic_dim: int = 768     # Language/semantic embedding
    intent_dim: int = 512       # Intent representation
    emotion_dim: int = 256      # Emotional state
    memory_dim: int = 512       # Memory encoding
    action_dim: int = 256       # Action representation
    

@dataclass
class ContextualInput:
    """Input context for decision making."""
    situation_encoding: torch.Tensor  # [perception_dim]
    target_profiles: List[torch.Tensor]  # List of [perception_dim] for each person
    social_context: torch.Tensor  # [semantic_dim]
    historical_context: torch.Tensor  # [memory_dim]
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalState:
    """AI's internal state representation."""
    emotional_state: torch.Tensor  # [emotion_dim]
    moral_alignment: torch.Tensor  # [emotion_dim]
    relationship_memory: Dict[str, torch.Tensor]  # person_id -> [memory_dim]
    recent_actions: List[ActionType] = field(default_factory=list)
    confidence_level: float = 0.5
    arousal_level: float = 0.0  # Internal "energy" level
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentDraft:
    """Draft intention before desire conflict resolution."""
    goal_embedding: torch.Tensor  # [intent_dim]
    priority: float  # 0.0 to 1.0
    action_type: ActionType
    target_ids: List[str] = field(default_factory=list)
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentResolved:
    """Resolved intention after desire conflict resolution."""
    action_type: ActionType
    action_embedding: torch.Tensor  # [action_dim]
    confidence: float
    moral_score: float  # -1.0 (harmful) to 1.0 (beneficial)
    expected_outcome: Optional['SimulationResult'] = None
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TheoryOfMindState:
    """Predicted mental state of another agent."""
    beliefs: torch.Tensor  # [semantic_dim]
    desires: torch.Tensor  # [intent_dim]
    emotions: torch.Tensor  # [emotion_dim]
    predicted_actions: torch.Tensor  # [action_dim]
    confidence: float = 0.5


@dataclass
class EmotionalImpact:
    """Predicted emotional impact of an action."""
    target_emotion_delta: torch.Tensor  # [emotion_dim] - change in target's emotions
    self_emotion_delta: torch.Tensor  # [emotion_dim] - change in AI's emotions
    valence: float  # -1.0 (negative) to 1.0 (positive)
    arousal: float  # 0.0 (calm) to 1.0 (excited)
    confidence: float = 0.5


@dataclass
class OutcomeProjection:
    """Predicted outcome of an action."""
    success_probability: float
    consequence_embedding: torch.Tensor  # [semantic_dim]
    time_horizon: float  # How far into future (in abstract units)
    side_effects: List[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class RelationshipTrajectory:
    """Predicted relationship evolution."""
    relationship_delta: torch.Tensor  # [memory_dim] - change in relationship state
    trust_delta: float  # -1.0 to 1.0
    affection_delta: float  # -1.0 to 1.0
    bond_strength: float  # 0.0 to 1.0
    time_horizon: float
    confidence: float = 0.5


@dataclass
class SimulationResult:
    """Aggregate simulation results for an action."""
    theory_of_mind: Optional[TheoryOfMindState] = None
    emotional_impact: Optional[EmotionalImpact] = None
    outcome_projection: Optional[OutcomeProjection] = None
    relationship_trajectory: Optional[RelationshipTrajectory] = None
    overall_score: float = 0.0  # Weighted combination of all factors
    confidence: float = 0.5


@dataclass
class DecisionLogits:
    """Raw decision logits before final selection."""
    action_logits: torch.Tensor  # [num_actions]
    moral_logits: torch.Tensor  # [2] - kind vs mean tendency
    confidence_logit: float
    temperature: float = 1.0
    

@dataclass
class DecisionTrace:
    """Metadata tracking decision rationale."""
    intent_draft: IntentDraft
    intent_resolved: IntentResolved
    simulation: SimulationResult
    final_action: ActionType
    decision_logits: DecisionLogits
    expert_activations: Dict[str, float]  # Which experts were activated
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
