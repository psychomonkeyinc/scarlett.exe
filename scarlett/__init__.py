"""Scarlett AI - A comprehensive AI decision-making system."""

from .scarlett import ScarlettAI, create_scarlett_ai
from .core.config import (
    ScarlettConfig,
    ModelScale,
    SafetyConfig,
    TrainingConfig,
    get_default_config,
    AVAILABLE_SCALES
)
from .core.types import (
    ActionType,
    ContextualInput,
    InternalState,
    IntentDraft,
    IntentResolved,
    SimulationResult,
    DecisionTrace,
    DecisionLogits
)

__version__ = "0.1.0"

__all__ = [
    # Main system
    'ScarlettAI',
    'create_scarlett_ai',
    
    # Configuration
    'ScarlettConfig',
    'ModelScale',
    'SafetyConfig',
    'TrainingConfig',
    'get_default_config',
    'AVAILABLE_SCALES',
    
    # Types
    'ActionType',
    'ContextualInput',
    'InternalState',
    'IntentDraft',
    'IntentResolved',
    'SimulationResult',
    'DecisionTrace',
    'DecisionLogits',
]
