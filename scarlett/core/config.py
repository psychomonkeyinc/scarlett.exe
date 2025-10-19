"""Configuration system for Scarlett AI.

Provides scalable configuration from 500M to 30B+ parameters.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import os


@dataclass
class ModelScale:
    """Model scale configuration - determines total parameter count."""
    name: str
    total_params: str  # e.g., "500M", "2B", "7B", "30B"
    
    # Core dimensions
    hidden_dim: int
    num_layers: int
    num_heads: int
    ff_dim: int
    
    # Expert configuration (for Mixture of Experts)
    num_experts: int
    expert_capacity: int
    active_experts: int  # How many experts to activate per token
    
    # Perception
    vision_dim: int
    audio_dim: int
    multimodal_dim: int
    
    # Intent and decision
    intent_dim: int
    action_dim: int
    
    # Memory
    memory_dim: int
    max_memory_items: int
    
    # Simulation
    simulation_depth: int  # How many steps to simulate ahead
    

# Predefined model scales
SCALE_500M = ModelScale(
    name="500M",
    total_params="500M",
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    ff_dim=3072,
    num_experts=8,
    expert_capacity=64,
    active_experts=2,
    vision_dim=512,
    audio_dim=256,
    multimodal_dim=1024,
    intent_dim=512,
    action_dim=256,
    memory_dim=512,
    max_memory_items=1000,
    simulation_depth=3,
)

SCALE_2B = ModelScale(
    name="2B",
    total_params="2B",
    hidden_dim=1536,
    num_layers=24,
    num_heads=16,
    ff_dim=6144,
    num_experts=16,
    expert_capacity=128,
    active_experts=4,
    vision_dim=1024,
    audio_dim=512,
    multimodal_dim=2048,
    intent_dim=1024,
    action_dim=512,
    memory_dim=1024,
    max_memory_items=5000,
    simulation_depth=5,
)

SCALE_7B = ModelScale(
    name="7B",
    total_params="7B",
    hidden_dim=4096,
    num_layers=32,
    num_heads=32,
    ff_dim=16384,
    num_experts=32,
    expert_capacity=256,
    active_experts=8,
    vision_dim=2048,
    audio_dim=1024,
    multimodal_dim=4096,
    intent_dim=2048,
    action_dim=1024,
    memory_dim=2048,
    max_memory_items=20000,
    simulation_depth=7,
)

SCALE_30B = ModelScale(
    name="30B",
    total_params="30B",
    hidden_dim=8192,
    num_layers=48,
    num_heads=64,
    ff_dim=32768,
    num_experts=64,
    expert_capacity=512,
    active_experts=16,
    vision_dim=4096,
    audio_dim=2048,
    multimodal_dim=8192,
    intent_dim=4096,
    action_dim=2048,
    memory_dim=4096,
    max_memory_items=100000,
    simulation_depth=10,
)

AVAILABLE_SCALES = {
    "500M": SCALE_500M,
    "2B": SCALE_2B,
    "7B": SCALE_7B,
    "30B": SCALE_30B,
}


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Ethical training
    ethical_reward_weight: float = 0.5
    consistency_weight: float = 0.1
    diversity_weight: float = 0.05
    

@dataclass
class SafetyConfig:
    """Safety and guardrail configuration."""
    enable_guardrails: bool = True
    max_harm_score: float = -0.5  # Actions with harm < this are blocked
    require_moral_justification: bool = True
    log_all_decisions: bool = True
    enable_self_testing: bool = True
    self_test_frequency: int = 1000  # Every N decisions
    

@dataclass
class ScarlettConfig:
    """Main configuration for Scarlett AI."""
    model_scale: ModelScale
    training: TrainingConfig = field(default_factory=TrainingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    
    # Runtime
    device: str = "cuda"
    seed: int = 42
    log_level: str = "INFO"
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    data_dir: str = "./data"
    
    def __post_init__(self):
        """Initialize derived configurations."""
        # Adjust training config based on model scale
        if self.model_scale.name in ["7B", "30B"]:
            self.training.gradient_checkpointing = True
            self.training.mixed_precision = True
    
    @classmethod
    def from_scale(cls, scale_name: str, **kwargs) -> 'ScarlettConfig':
        """Create config from a predefined scale."""
        if scale_name not in AVAILABLE_SCALES:
            raise ValueError(
                f"Unknown scale {scale_name}. Available: {list(AVAILABLE_SCALES.keys())}"
            )
        model_scale = AVAILABLE_SCALES[scale_name]
        return cls(model_scale=model_scale, **kwargs)
    
    @classmethod
    def from_file(cls, path: str) -> 'ScarlettConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Parse model scale
        scale_name = data.pop('model_scale', '500M')
        model_scale = AVAILABLE_SCALES.get(scale_name, SCALE_500M)
        
        # Parse nested configs
        training = TrainingConfig(**data.pop('training', {}))
        safety = SafetyConfig(**data.pop('safety', {}))
        
        return cls(
            model_scale=model_scale,
            training=training,
            safety=safety,
            **data
        )
    
    def to_file(self, path: str):
        """Save configuration to JSON file."""
        data = {
            'model_scale': self.model_scale.name,
            'training': {
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'warmup_steps': self.training.warmup_steps,
                'max_steps': self.training.max_steps,
                'gradient_accumulation_steps': self.training.gradient_accumulation_steps,
                'mixed_precision': self.training.mixed_precision,
                'gradient_checkpointing': self.training.gradient_checkpointing,
                'ethical_reward_weight': self.training.ethical_reward_weight,
                'consistency_weight': self.training.consistency_weight,
                'diversity_weight': self.training.diversity_weight,
            },
            'safety': {
                'enable_guardrails': self.safety.enable_guardrails,
                'max_harm_score': self.safety.max_harm_score,
                'require_moral_justification': self.safety.require_moral_justification,
                'log_all_decisions': self.safety.log_all_decisions,
                'enable_self_testing': self.safety.enable_self_testing,
                'self_test_frequency': self.safety.self_test_frequency,
            },
            'device': self.device,
            'seed': self.seed,
            'log_level': self.log_level,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'data_dir': self.data_dir,
        }
        
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# Default configuration
def get_default_config(scale: str = "500M") -> ScarlettConfig:
    """Get default configuration for a given scale."""
    return ScarlettConfig.from_scale(scale)
