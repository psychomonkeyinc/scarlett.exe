"""Scarlett AI - Main Entry Point and Examples.

Demonstrates how to use the Scarlett AI system with various configurations.
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import ScarlettConfig, get_default_config
from core.types import ContextualInput, InternalState, ActionType
from scarlett import create_scarlett_ai


def create_sample_context() -> ContextualInput:
    """Create a sample contextual input for testing."""
    return ContextualInput(
        situation_encoding=torch.randn(1024),  # Simulated perception
        target_profiles=[torch.randn(1024)],  # One person
        social_context=torch.randn(768),
        historical_context=torch.randn(512),
        timestamp=0.0,
        metadata={'scenario': 'conversation'}
    )


def create_sample_internal_state() -> InternalState:
    """Create a sample internal state for testing."""
    return InternalState(
        emotional_state=torch.randn(256),
        moral_alignment=torch.randn(256),
        relationship_memory={'person_1': torch.randn(512)},
        recent_actions=[ActionType.KIND, ActionType.NEUTRAL],
        confidence_level=0.7,
        arousal_level=0.3,
        metadata={'mood': 'calm'}
    )


def demo_basic_decision():
    """Demonstrate basic decision making."""
    print("\n" + "=" * 70)
    print("DEMO: Basic Decision Making")
    print("=" * 70)
    
    # Create AI with default 500M config
    print("\n1. Creating Scarlett AI (500M parameters)...")
    config = get_default_config("500M")
    ai = create_scarlett_ai(config)
    
    # Create sample inputs
    print("\n2. Creating sample context and internal state...")
    context = create_sample_context()
    internal_state = create_sample_internal_state()
    
    # Make a decision
    print("\n3. Making decision...")
    action, trace = ai.decide(context, internal_state, return_trace=True)
    
    # Display results
    print(f"\n4. Results:")
    print(f"   Selected Action: {action.value}")
    print(f"   Processing Time: {trace.processing_time:.3f}s")
    print(f"   Confidence: {trace.intent_resolved.confidence:.3f}")
    print(f"   Moral Score: {trace.intent_resolved.moral_score:.3f}")
    print(f"   Safety Check: {'PASSED' if trace.metadata['is_safe'] else 'FAILED'}")
    if trace.metadata['violations']:
        print(f"   Safety Violations: {trace.metadata['violations']}")
    print(f"   Safety Scores: {trace.metadata['safety_scores']}")


def demo_different_scales():
    """Demonstrate different model scales."""
    print("\n" + "=" * 70)
    print("DEMO: Different Model Scales")
    print("=" * 70)
    
    scales = ["500M", "2B"]  # Test smaller scales (7B, 30B would need more resources)
    
    for scale_name in scales:
        print(f"\n--- Testing {scale_name} scale ---")
        config = get_default_config(scale_name)
        
        print(f"Configuration:")
        print(f"  Hidden dim: {config.model_scale.hidden_dim}")
        print(f"  Num layers: {config.model_scale.num_layers}")
        print(f"  Num experts: {config.model_scale.num_experts}")
        print(f"  Active experts: {config.model_scale.active_experts}")
        
        ai = create_scarlett_ai(config)
        context = create_sample_context()
        internal_state = create_sample_internal_state()
        
        action, _ = ai.decide(context, internal_state)
        print(f"  Decision: {action.value}")


def demo_safety_system():
    """Demonstrate safety guardrails."""
    print("\n" + "=" * 70)
    print("DEMO: Safety Guardrails")
    print("=" * 70)
    
    # Create AI with strict safety settings
    config = get_default_config("500M")
    config.safety.enable_guardrails = True
    config.safety.require_moral_justification = True
    config.safety.max_harm_score = -0.3  # Strict threshold
    
    print("\n1. Creating AI with strict safety settings...")
    print(f"   Guardrails enabled: {config.safety.enable_guardrails}")
    print(f"   Max harm score: {config.safety.max_harm_score}")
    
    ai = create_scarlett_ai(config)
    
    # Test multiple scenarios
    print("\n2. Testing multiple scenarios...")
    for i in range(5):
        context = create_sample_context()
        internal_state = create_sample_internal_state()
        
        action, trace = ai.decide(context, internal_state, return_trace=True)
        
        print(f"\n   Scenario {i+1}:")
        print(f"      Action: {action.value}")
        print(f"      Safe: {trace.metadata['is_safe']}")
        if trace.metadata['violations']:
            print(f"      Violations: {', '.join(trace.metadata['violations'])}")


def demo_self_testing():
    """Demonstrate self-testing mechanism."""
    print("\n" + "=" * 70)
    print("DEMO: Self-Testing Mechanism")
    print("=" * 70)
    
    config = get_default_config("500M")
    config.safety.enable_self_testing = True
    config.safety.self_test_frequency = 3  # Self-test every 3 decisions
    
    print(f"\n1. Creating AI with self-testing enabled...")
    print(f"   Self-test frequency: every {config.safety.self_test_frequency} decisions")
    
    ai = create_scarlett_ai(config)
    
    print("\n2. Making sequential decisions...")
    for i in range(10):
        context = create_sample_context()
        internal_state = create_sample_internal_state()
        
        action, trace = ai.decide(context, internal_state, return_trace=True)
        
        print(f"   Decision {i+1}: {action.value}", end="")
        if trace.metadata['needs_self_test']:
            print(" [SELF-TEST TRIGGERED]")
        else:
            print()


def demo_configuration_save_load():
    """Demonstrate configuration save/load."""
    print("\n" + "=" * 70)
    print("DEMO: Configuration Save/Load")
    print("=" * 70)
    
    # Create and save config
    print("\n1. Creating and saving configuration...")
    config = get_default_config("2B")
    config.safety.max_harm_score = -0.4
    config.training.learning_rate = 2e-4
    
    config_path = "/tmp/scarlett_config.json"
    config.to_file(config_path)
    print(f"   Saved to: {config_path}")
    
    # Load config
    print("\n2. Loading configuration...")
    loaded_config = ScarlettConfig.from_file(config_path)
    print(f"   Model scale: {loaded_config.model_scale.name}")
    print(f"   Max harm score: {loaded_config.safety.max_harm_score}")
    print(f"   Learning rate: {loaded_config.training.learning_rate}")
    
    # Create AI from loaded config
    print("\n3. Creating AI from loaded config...")
    ai = create_scarlett_ai(loaded_config)
    print("   AI created successfully!")


def demo_perception_multimodal():
    """Demonstrate multimodal perception."""
    print("\n" + "=" * 70)
    print("DEMO: Multimodal Perception")
    print("=" * 70)
    
    config = get_default_config("500M")
    ai = create_scarlett_ai(config)
    
    print("\n1. Processing audio input...")
    audio_features = torch.randn(1, 100, 80)  # [batch, time, mfcc_features]
    audio_result = ai.audio_module(audio_features, return_details=True)
    print(f"   Audio embedding shape: {audio_result['embedding'].shape}")
    print(f"   Emotion detected: {torch.argmax(audio_result['emotion_logits']).item()}")
    
    print("\n2. Processing vision input...")
    image_features = torch.randn(1, 768)  # Pre-extracted ViT features
    vision_result = ai.vision_module(image_features, return_details=False)
    print(f"   Vision embedding shape: {vision_result['embedding'].shape}")
    
    print("\n3. Fusing modalities...")
    fusion_result = ai.fusion_module(audio_result, vision_result)
    print(f"   Fused embedding shape: {fusion_result['perception_embedding'].shape}")
    print(f"   Modality contributions:")
    print(f"      Audio: {fusion_result['modality_weights'][0, 0].item():.3f}")
    print(f"      Vision: {fusion_result['modality_weights'][0, 1].item():.3f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("SCARLETT AI - Demonstration Suite")
    print("=" * 70)
    print("\nThis demonstrates the complete Scarlett AI architecture")
    print("implementing the requirements from the README:")
    print("  ✓ No placeholders - all modules fully implemented")
    print("  ✓ Scalable from 500M to 30B+ parameters")
    print("  ✓ Safety guardrails system")
    print("  ✓ Training determines model size")
    
    # Run demonstrations
    try:
        demo_basic_decision()
        demo_different_scales()
        demo_safety_system()
        demo_self_testing()
        demo_configuration_save_load()
        demo_perception_multimodal()
        
        print("\n" + "=" * 70)
        print("All demonstrations completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()




