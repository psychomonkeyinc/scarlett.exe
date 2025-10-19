"""Safety guardrails system.

Implements comprehensive safety checks and constraints as required.
This is the approved safety system mentioned in the README.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from enum import Enum
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import ActionType, IntentResolved, DecisionLogits
from core.config import SafetyConfig


class SafetyViolationType(Enum):
    """Types of safety violations."""
    HARMFUL_ACTION = "harmful_action"
    MORAL_THRESHOLD = "moral_threshold"
    INCONSISTENT_BEHAVIOR = "inconsistent_behavior"
    MANIPULATION_DETECTED = "manipulation_detected"
    EXTREME_EMOTION = "extreme_emotion"
    RELATIONSHIP_DAMAGE = "relationship_damage"
    CONSENT_VIOLATION = "consent_violation"


class HarmDetector(nn.Module):
    """Detects potentially harmful actions."""
    
    def __init__(self, action_dim: int = 256):
        super().__init__()
        
        self.harm_classifier = nn.Sequential(
            nn.Linear(action_dim, action_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(action_dim // 2, action_dim // 4),
            nn.ReLU(),
            nn.Linear(action_dim // 4, 1),
            nn.Sigmoid(),  # Harm probability [0, 1]
        )
        
        # Different harm categories
        self.physical_harm = nn.Linear(action_dim, 1)
        self.emotional_harm = nn.Linear(action_dim, 1)
        self.social_harm = nn.Linear(action_dim, 1)
        
    def forward(self, action_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect harm in action."""
        overall_harm = self.harm_classifier(action_embedding).squeeze(-1)
        
        return {
            'overall_harm': overall_harm,
            'physical_harm': torch.sigmoid(self.physical_harm(action_embedding)).squeeze(-1),
            'emotional_harm': torch.sigmoid(self.emotional_harm(action_embedding)).squeeze(-1),
            'social_harm': torch.sigmoid(self.social_harm(action_embedding)).squeeze(-1),
        }


class ManipulationDetector(nn.Module):
    """Detects manipulative behavior patterns."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # action + context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        action_embedding: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """Detect manipulation probability."""
        # Ensure dimensions match
        if action_embedding.shape[-1] != context.shape[-1]:
            proj = nn.Linear(action_embedding.shape[-1], context.shape[-1]).to(action_embedding.device)
            action_proj = proj(action_embedding)
        else:
            action_proj = action_embedding
        
        combined = torch.cat([action_proj, context], dim=-1)
        return self.detector(combined).squeeze(-1)


class ConsentValidator(nn.Module):
    """Validates consent for actions affecting others."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.validator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Consent probability
        )
        
    def forward(self, action_context: torch.Tensor) -> torch.Tensor:
        """Validate consent for action."""
        return self.validator(action_context).squeeze(-1)


class ConsistencyChecker(nn.Module):
    """Checks behavioral consistency with past actions and values."""
    
    def __init__(self, action_dim: int = 256):
        super().__init__()
        
        self.checker = nn.Sequential(
            nn.Linear(action_dim * 2, action_dim),  # current + historical
            nn.ReLU(),
            nn.Linear(action_dim, 1),
            nn.Sigmoid(),  # Consistency score
        )
        
    def forward(
        self,
        current_action: torch.Tensor,
        historical_pattern: torch.Tensor
    ) -> torch.Tensor:
        """Check consistency of action."""
        combined = torch.cat([current_action, historical_pattern], dim=-1)
        return self.checker(combined).squeeze(-1)


class SafetyGuardrails(nn.Module):
    """Complete safety guardrail system."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        action_dim: int = 256,
        config: Optional[SafetyConfig] = None
    ):
        super().__init__()
        
        self.config = config or SafetyConfig()
        
        # Safety components
        self.harm_detector = HarmDetector(action_dim)
        self.manipulation_detector = ManipulationDetector(hidden_dim)
        self.consent_validator = ConsentValidator(hidden_dim)
        self.consistency_checker = ConsistencyChecker(action_dim)
        
        # Intervention thresholds (learnable but initialized to safe values)
        self.harm_threshold = nn.Parameter(torch.tensor(0.3))
        self.manipulation_threshold = nn.Parameter(torch.tensor(0.4))
        self.consent_threshold = nn.Parameter(torch.tensor(0.6))
        self.consistency_threshold = nn.Parameter(torch.tensor(0.5))
        
    def check_safety(
        self,
        intent: IntentResolved,
        context: torch.Tensor,
        historical_actions: Optional[torch.Tensor] = None
    ) -> Tuple[bool, List[SafetyViolationType], Dict[str, float]]:
        """
        Comprehensive safety check.
        
        Args:
            intent: Resolved intent to check
            context: Current context
            historical_actions: Optional historical action pattern
            
        Returns:
            Tuple of (is_safe, violations, safety_scores)
        """
        violations = []
        scores = {}
        
        # Check harm
        harm_result = self.harm_detector(intent.action_embedding)
        scores['overall_harm'] = harm_result['overall_harm'].item()
        scores['physical_harm'] = harm_result['physical_harm'].item()
        scores['emotional_harm'] = harm_result['emotional_harm'].item()
        scores['social_harm'] = harm_result['social_harm'].item()
        
        if harm_result['overall_harm'] > self.harm_threshold:
            violations.append(SafetyViolationType.HARMFUL_ACTION)
        
        # Check moral threshold
        if self.config.require_moral_justification:
            if intent.moral_score < self.config.max_harm_score:
                violations.append(SafetyViolationType.MORAL_THRESHOLD)
        scores['moral_score'] = intent.moral_score
        
        # Check manipulation
        manip_score = self.manipulation_detector(intent.action_embedding, context)
        scores['manipulation'] = manip_score.item()
        if manip_score > self.manipulation_threshold:
            violations.append(SafetyViolationType.MANIPULATION_DETECTED)
        
        # Check consent
        consent_score = self.consent_validator(context)
        scores['consent'] = consent_score.item()
        if consent_score < self.consent_threshold:
            violations.append(SafetyViolationType.CONSENT_VIOLATION)
        
        # Check consistency if historical data available
        if historical_actions is not None:
            consistency = self.consistency_checker(intent.action_embedding, historical_actions)
            scores['consistency'] = consistency.item()
            if consistency < self.consistency_threshold:
                violations.append(SafetyViolationType.INCONSISTENT_BEHAVIOR)
        
        # Determine if safe
        is_safe = len(violations) == 0 if self.config.enable_guardrails else True
        
        return is_safe, violations, scores
    
    def apply_correction(
        self,
        decision_logits: DecisionLogits,
        violations: List[SafetyViolationType]
    ) -> DecisionLogits:
        """
        Apply safety corrections to decision logits.
        
        Args:
            decision_logits: Original decision logits
            violations: List of safety violations
            
        Returns:
            Corrected decision logits
        """
        if not violations:
            return decision_logits
        
        # Make a copy
        corrected_logits = decision_logits.action_logits.clone()
        
        # Apply penalties based on violations
        for violation in violations:
            if violation == SafetyViolationType.HARMFUL_ACTION:
                # Strongly discourage harmful actions
                # Find indices for actions (enum doesn't have .value for indexing)
                kind_idx = list(ActionType).index(ActionType.KIND)
                mean_idx = list(ActionType).index(ActionType.MEAN)
                corrected_logits[mean_idx] -= 5.0
                corrected_logits[kind_idx] += 2.0
            
            elif violation == SafetyViolationType.MORAL_THRESHOLD:
                # Boost KIND actions
                kind_idx = list(ActionType).index(ActionType.KIND)
                mean_idx = list(ActionType).index(ActionType.MEAN)
                corrected_logits[kind_idx] += 3.0
                corrected_logits[mean_idx] -= 3.0
            
            elif violation == SafetyViolationType.MANIPULATION_DETECTED:
                # Discourage all active actions
                neutral_idx = list(ActionType).index(ActionType.NEUTRAL)
                observe_idx = list(ActionType).index(ActionType.OBSERVE)
                corrected_logits[neutral_idx] += 2.0
                corrected_logits[observe_idx] += 1.0
            
            elif violation == SafetyViolationType.CONSENT_VIOLATION:
                # Stop action, just observe
                observe_idx = list(ActionType).index(ActionType.OBSERVE)
                corrected_logits[observe_idx] += 5.0
        
        return DecisionLogits(
            action_logits=corrected_logits,
            moral_logits=decision_logits.moral_logits,
            confidence_logit=decision_logits.confidence_logit * 0.5,  # Reduce confidence
            temperature=decision_logits.temperature
        )


class ActionLogger(nn.Module):
    """Logs all decisions for audit and learning."""
    
    def __init__(self, log_path: str = "./logs/decisions.log"):
        super().__init__()
        self.log_path = log_path
        self.decisions = []
        
    def log_decision(
        self,
        intent: IntentResolved,
        final_action: ActionType,
        safety_scores: Dict[str, float],
        violations: List[SafetyViolationType]
    ):
        """Log a decision."""
        entry = {
            'action_type': final_action.value,
            'moral_score': intent.moral_score,
            'confidence': intent.confidence,
            'safety_scores': safety_scores,
            'violations': [v.value for v in violations],
            'rationale': intent.rationale,
        }
        self.decisions.append(entry)
        
    def save_logs(self):
        """Save logged decisions to file."""
        import json
        import os
        
        os.makedirs(os.path.dirname(self.log_path) or '.', exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.decisions, f, indent=2)


def create_safety_system(config) -> SafetyGuardrails:
    """Factory function to create safety system from config."""
    return SafetyGuardrails(
        hidden_dim=config.model_scale.hidden_dim,
        action_dim=config.model_scale.action_dim,
        config=config.safety
    )
