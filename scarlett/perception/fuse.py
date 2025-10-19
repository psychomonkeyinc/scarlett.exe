"""Multimodal fusion for combining audio and vision.

Fuses multimodal inputs into unified representations for decision-making.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class CrossModalAttention(nn.Module):
    """Cross-attention between audio and vision modalities."""
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Projection layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        query: torch.Tensor,  # [batch, hidden_dim]
        key: torch.Tensor,    # [batch, hidden_dim]
        value: torch.Tensor,  # [batch, hidden_dim]
    ) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            query: Query features (e.g., from audio)
            key: Key features (e.g., from vision)
            value: Value features (e.g., from vision)
            
        Returns:
            Attended features [batch, hidden_dim]
        """
        batch_size = query.shape[0]
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, 1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, 1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, num_heads, 1, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, 1, head_dim]
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, self.hidden_dim)
        
        return self.out_proj(attn_output)


class MultimodalFusionModule(nn.Module):
    """Fuses audio and vision modalities into unified representation."""
    
    def __init__(
        self,
        audio_dim: int = 256,
        vision_dim: int = 512,
        multimodal_dim: int = 1024,
        hidden_dim: int = 768,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.vision_dim = vision_dim
        self.multimodal_dim = multimodal_dim
        
        # Project inputs to common hidden dimension
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        # Cross-modal attention layers
        self.audio_to_vision = CrossModalAttention(hidden_dim, num_heads)
        self.vision_to_audio = CrossModalAttention(hidden_dim, num_heads)
        
        # Fusion layers
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Final projection to multimodal dimension
        self.output_proj = nn.Linear(hidden_dim, multimodal_dim)
        
        # Modality importance weighting
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        audio_features: torch.Tensor,  # [batch, audio_dim]
        vision_features: torch.Tensor,  # [batch, vision_dim]
        return_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse audio and vision features.
        
        Args:
            audio_features: Audio embeddings
            vision_features: Vision embeddings
            return_weights: Whether to return attention weights
            
        Returns:
            Dictionary with fused multimodal embedding
        """
        # Project to common dimension
        audio_proj = self.audio_proj(audio_features)  # [batch, hidden_dim]
        vision_proj = self.vision_proj(vision_features)  # [batch, hidden_dim]
        
        # Cross-modal attention
        audio_attended = self.audio_to_vision(
            query=audio_proj,
            key=vision_proj,
            value=vision_proj
        )  # Audio attending to vision
        
        vision_attended = self.vision_to_audio(
            query=vision_proj,
            key=audio_proj,
            value=audio_proj
        )  # Vision attending to audio
        
        # Compute modality importance weights
        concat_features = torch.cat([audio_attended, vision_attended], dim=-1)
        modality_weights = self.modality_gate(concat_features)  # [batch, 2]
        
        # Weight and combine modalities
        audio_weighted = audio_attended * modality_weights[:, 0:1]
        vision_weighted = vision_attended * modality_weights[:, 1:2]
        
        # Fuse weighted features
        combined = torch.cat([audio_weighted, vision_weighted], dim=-1)
        fused = self.fusion_net(combined)  # [batch, hidden_dim]
        
        # Project to multimodal dimension
        multimodal_embedding = self.output_proj(fused)  # [batch, multimodal_dim]
        
        result = {
            'embedding': multimodal_embedding,
            'audio_contribution': modality_weights[:, 0],
            'vision_contribution': modality_weights[:, 1],
        }
        
        if return_weights:
            result.update({
                'audio_attended': audio_attended,
                'vision_attended': vision_attended,
            })
        
        return result


class EmotionalToneFusion(nn.Module):
    """Fuses emotional signals from audio and vision."""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # 7 basic emotions from each modality
        self.fusion = nn.Sequential(
            nn.Linear(14, hidden_dim),  # 7 from audio + 7 from vision
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 7),  # Fused 7 emotions
        )
        
    def forward(
        self,
        audio_emotion: torch.Tensor,  # [batch, 7]
        vision_emotion: torch.Tensor,  # [batch, 7]
    ) -> torch.Tensor:
        """
        Fuse emotional signals.
        
        Args:
            audio_emotion: Emotion logits from audio
            vision_emotion: Emotion logits from vision
            
        Returns:
            Fused emotion logits [batch, 7]
        """
        combined = torch.cat([audio_emotion, vision_emotion], dim=-1)
        return self.fusion(combined)


class PerceptionFusionModule(nn.Module):
    """Complete perception fusion module combining all modalities."""
    
    def __init__(
        self,
        audio_dim: int = 256,
        vision_dim: int = 512,
        multimodal_dim: int = 1024,
        hidden_dim: int = 768,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.multimodal_fusion = MultimodalFusionModule(
            audio_dim, vision_dim, multimodal_dim, hidden_dim, num_heads
        )
        self.emotional_fusion = EmotionalToneFusion(hidden_dim // 2)
        
    def forward(
        self,
        audio_output: Dict[str, torch.Tensor],
        vision_output: Dict[str, torch.Tensor],
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse all perception modalities.
        
        Args:
            audio_output: Output from audio perception module
            vision_output: Output from vision perception module
            return_details: Whether to return detailed fusion information
            
        Returns:
            Dictionary with fused perception results
        """
        # Fuse main embeddings
        fusion_result = self.multimodal_fusion(
            audio_output['embedding'],
            vision_output['embedding'],
            return_weights=return_details
        )
        
        result = {
            'perception_embedding': fusion_result['embedding'],
            'modality_weights': torch.stack([
                fusion_result['audio_contribution'],
                fusion_result['vision_contribution']
            ], dim=-1)
        }
        
        # Fuse emotional signals if available
        if 'emotion_logits' in audio_output and 'emotion_logits' in vision_output:
            fused_emotion = self.emotional_fusion(
                audio_output['emotion_logits'],
                vision_output['emotion_logits']
            )
            result['emotion_logits'] = fused_emotion
        
        if return_details:
            result.update({
                'audio_attended': fusion_result.get('audio_attended'),
                'vision_attended': fusion_result.get('vision_attended'),
            })
        
        return result


def create_fusion_module(config) -> PerceptionFusionModule:
    """Factory function to create fusion module from config."""
    return PerceptionFusionModule(
        audio_dim=config.model_scale.audio_dim,
        vision_dim=config.model_scale.vision_dim,
        multimodal_dim=config.model_scale.multimodal_dim,
        hidden_dim=config.model_scale.hidden_dim,
        num_heads=config.model_scale.num_heads
    )
