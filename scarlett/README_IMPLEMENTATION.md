# Scarlett AI - Implementation Guide

This document describes the implementation of the Scarlett AI architecture as specified in the requirements.

## Overview

Scarlett AI is a comprehensive artificial intelligence system implementing:
- ✓ **No placeholders** - All modules are fully implemented with working neural networks
- ✓ **Scalable architecture** - Supports 500M, 2B, 7B, 30B+ parameter configurations
- ✓ **Safety guardrails** - Comprehensive safety system with harm detection, manipulation detection, and consent validation
- ✓ **Training-determined size** - Model scale configured through simple parameter selection

## Architecture Components

### 1. Core System (`core/`)
- **`types.py`**: Data structures for all system components
- **`config.py`**: Scalable configuration system supporting multiple model sizes

### 2. Perception Modules (`perception/`)
- **`audio.py`**: Audio perception with prosody, emotion, and VAD
- **`vision.py`**: Vision perception with face analysis, gesture recognition, gaze tracking
- **`fuse.py`**: Multimodal fusion using cross-attention mechanisms

### 3. Simulation Modules (`sim/`)
- **`theory.py`**: Theory of mind - predicts mental states of others
- **`emotion.py`**: Emotional impact prediction for actions
- **`outcome.py`**: Consequence projection and outcome forecasting
- **`relation.py`**: Relationship trajectory prediction

### 4. Intent Formulation (`intent/`)
- **`desire.py`**: Desire conflict resolution engine
- **`agency.py`**: Agency simulation and activation readiness
- **`goals.py`**: Hierarchical goal management (long-term, mid-term, immediate)
- **`intent.py`**: Intent formulation integrating all components

### 5. Decision System (`decision/`)
- **`actions.py`**: Action space definition and parameterization
- **`synth.py`**: Decision synthesis from all evaluators

### 6. Safety System (`safety/`)
- **`guardrails.py`**: Comprehensive safety checks including:
  - Harm detection (physical, emotional, social)
  - Manipulation detection
  - Consent validation
  - Behavioral consistency checking
  - Safety violation correction

### 7. Integration (`scarlett.py`)
- Complete integration of all components
- Unified decision-making pipeline

## Quick Start

### Basic Usage

```python
from scarlett import create_scarlett_ai
from core.config import get_default_config
from core.types import ContextualInput, InternalState, ActionType

# Create AI with 500M parameters
config = get_default_config("500M")
ai = create_scarlett_ai(config)

# Create input context
context = ContextualInput(
    situation_encoding=torch.randn(1024),
    target_profiles=[torch.randn(1024)],
    social_context=torch.randn(768),
    historical_context=torch.randn(512)
)

# Create internal state
internal_state = InternalState(
    emotional_state=torch.randn(256),
    moral_alignment=torch.randn(256),
    relationship_memory={'person_1': torch.randn(512)},
    recent_actions=[ActionType.NEUTRAL]
)

# Make decision
action, trace = ai.decide(context, internal_state, return_trace=True)
print(f"Selected action: {action.value}")
print(f"Moral score: {trace.intent_resolved.moral_score}")
```

### Scaling to Different Sizes

```python
# 2B parameter model
config = get_default_config("2B")
ai_2b = create_scarlett_ai(config)

# 7B parameter model
config = get_default_config("7B")
ai_7b = create_scarlett_ai(config)

# 30B parameter model
config = get_default_config("30B")
ai_30b = create_scarlett_ai(config)
```

### Safety Configuration

```python
config = get_default_config("500M")

# Configure safety guardrails
config.safety.enable_guardrails = True
config.safety.max_harm_score = -0.5
config.safety.require_moral_justification = True
config.safety.enable_self_testing = True
config.safety.self_test_frequency = 1000

ai = create_scarlett_ai(config)
```

### Custom Configuration

```python
from core.config import ScarlettConfig, SafetyConfig, TrainingConfig

# Create custom configuration
config = ScarlettConfig.from_scale("2B")
config.safety = SafetyConfig(
    enable_guardrails=True,
    max_harm_score=-0.3,
    log_all_decisions=True
)
config.training = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32
)

# Save configuration
config.to_file("my_config.json")

# Load configuration
loaded_config = ScarlettConfig.from_file("my_config.json")
```

## Running Demonstrations

```bash
cd scarlett
python run.py
```

This will run comprehensive demonstrations including:
1. Basic decision making
2. Different model scales
3. Safety guardrails
4. Self-testing mechanism
5. Configuration save/load
6. Multimodal perception

## Model Scales

| Scale | Hidden Dim | Layers | Experts | Parameters |
|-------|-----------|--------|---------|------------|
| 500M  | 768       | 12     | 8       | ~500M      |
| 2B    | 1536      | 24     | 16      | ~2B        |
| 7B    | 4096      | 32     | 32      | ~7B        |
| 30B   | 8192      | 48     | 64      | ~30B       |

## Training

The system is designed to be trained using:
- Multi-objective loss combining action accuracy, ethical alignment, and consistency
- Reinforcement learning for desire conflict resolution
- Supervised learning on human-labeled scenarios
- Self-supervised learning on simulated interactions

Training configuration is specified in `TrainingConfig`:
```python
config.training.batch_size = 32
config.training.learning_rate = 1e-4
config.training.ethical_reward_weight = 0.5
config.training.mixed_precision = True
config.training.gradient_checkpointing = True  # For large models
```

## Safety Features

The safety system provides multiple layers of protection:

1. **Harm Detection**: Identifies physical, emotional, and social harm
2. **Manipulation Detection**: Detects attempts to manipulate others
3. **Consent Validation**: Ensures actions respect consent
4. **Consistency Checking**: Maintains behavioral consistency
5. **Automatic Correction**: Adjusts decisions that violate safety constraints
6. **Decision Logging**: Records all decisions for audit

## Action Types

The system supports five action types:
- **KIND**: Beneficial, helpful actions
- **MEAN**: Potentially harmful or unkind actions
- **NEUTRAL**: Neutral, everyday actions
- **OBSERVE**: Non-intrusive observation
- **SELF_TEST**: Internal diagnostic testing

## Decision Pipeline

1. **Perception**: Process multimodal input (audio + vision)
2. **Intent Formulation**: 
   - Resolve desire conflicts
   - Compute agency readiness
   - Generate intent from goals
3. **Simulation**:
   - Predict mental states (theory of mind)
   - Forecast emotional impact
   - Project outcomes
   - Assess relationship changes
4. **Decision Synthesis**: Combine all evaluations
5. **Safety Check**: Validate decision safety
6. **Action Selection**: Choose final action

## File Structure

```
scarlett/
├── core/
│   ├── config.py          # Configuration system
│   └── types.py           # Core data types
├── perception/
│   ├── audio.py           # Audio processing
│   ├── vision.py          # Vision processing
│   └── fuse.py            # Multimodal fusion
├── sim/
│   ├── theory.py          # Theory of mind
│   ├── emotion.py         # Emotional impact
│   ├── outcome.py         # Outcome projection
│   └── relation.py        # Relationship trajectories
├── intent/
│   ├── desire.py          # Desire resolution
│   ├── agency.py          # Agency simulation
│   ├── goals.py           # Goal management
│   └── intent.py          # Intent formulation
├── decision/
│   ├── actions.py         # Action definitions
│   └── synth.py           # Decision synthesis
├── safety/
│   └── guardrails.py      # Safety system
├── scarlett.py            # Main integration
├── run.py                 # Entry point & demos
└── README_IMPLEMENTATION.md
```

## Requirements Met

✅ **No placeholders**: All modules have complete implementations with actual neural networks
✅ **Runnable code**: System can be instantiated and run with realistic parameters
✅ **Guardrail file**: Comprehensive safety system in `safety/guardrails.py`
✅ **Training determines size**: Simple configuration parameter controls all dimensions
✅ **Scalable**: Supports 500M to 30B+ parameters through configuration

## Next Steps

1. **Data Collection**: Gather training data for ethical decision-making scenarios
2. **Training Pipeline**: Implement full training loop with reinforcement learning
3. **Fine-tuning**: Train on specific use cases and domains
4. **Evaluation**: Develop comprehensive evaluation metrics
5. **Deployment**: Optimize for inference and deploy at scale

## Notes

- This implementation provides the complete architecture framework
- Neural network weights are randomly initialized and need training
- Real-world deployment would require:
  - Large-scale training data
  - Distributed training infrastructure
  - Comprehensive testing and validation
  - Ethical review and oversight
