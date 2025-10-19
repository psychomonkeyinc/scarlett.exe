"""Agency simulation module.

Models internal drive, activation readiness, and sense of autonomy.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ActivationReadinessNetwork(nn.Module):
    """Determines readiness to take action."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.readiness_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Readiness [0, 1]
        )
        
    def forward(self, internal_state: torch.Tensor) -> torch.Tensor:
        """Compute activation readiness."""
        return self.readiness_net(internal_state).squeeze(-1)


class InternalDriveNetwork(nn.Module):
    """Models internal motivational drive."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Multiple drive components
        self.achievement_drive = nn.Linear(hidden_dim, 1)
        self.social_drive = nn.Linear(hidden_dim, 1)
        self.exploration_drive = nn.Linear(hidden_dim, 1)
        self.maintenance_drive = nn.Linear(hidden_dim, 1)
        
        # Drive aggregation
        self.drive_combiner = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, internal_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute internal drives.
        
        Args:
            internal_state: Current internal state
            
        Returns:
            Dictionary of drive components and overall drive
        """
        # Individual drives
        achievement = torch.sigmoid(self.achievement_drive(internal_state))
        social = torch.sigmoid(self.social_drive(internal_state))
        exploration = torch.sigmoid(self.exploration_drive(internal_state))
        maintenance = torch.sigmoid(self.maintenance_drive(internal_state))
        
        # Combine drives
        all_drives = torch.cat([achievement, social, exploration, maintenance], dim=-1)
        overall_drive = self.drive_combiner(all_drives).squeeze(-1)
        
        return {
            'achievement': achievement.squeeze(-1),
            'social': social.squeeze(-1),
            'exploration': exploration.squeeze(-1),
            'maintenance': maintenance.squeeze(-1),
            'overall': overall_drive,
        }


class AutonomyNetwork(nn.Module):
    """Models sense of autonomy and self-determination."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.autonomy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Autonomy level [0, 1]
        )
        
        # Freedom of choice estimator
        self.freedom_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute autonomy and freedom metrics.
        
        Args:
            context: Current context
            
        Returns:
            Dictionary with autonomy metrics
        """
        autonomy = self.autonomy_net(context).squeeze(-1)
        freedom = self.freedom_net(context).squeeze(-1)
        
        return {
            'autonomy_level': autonomy,
            'freedom_of_choice': freedom,
        }


class ActionInitiationNetwork(nn.Module):
    """Determines whether to initiate action."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Combines drive, readiness, and context to decide on initiation
        self.initiation_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Initiation probability [0, 1]
        )
        
    def forward(
        self,
        drive_state: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute action initiation probability.
        
        Args:
            drive_state: Internal drive state
            context: Current context
            
        Returns:
            Initiation probability
        """
        combined = torch.cat([drive_state, context], dim=-1)
        return self.initiation_net(combined).squeeze(-1)


class AgencySimulationModule(nn.Module):
    """Complete agency simulation module."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Components
        self.readiness_network = ActivationReadinessNetwork(hidden_dim)
        self.drive_network = InternalDriveNetwork(hidden_dim)
        self.autonomy_network = AutonomyNetwork(hidden_dim)
        self.initiation_network = ActionInitiationNetwork(hidden_dim)
        
        # Internal state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
    def forward(
        self,
        internal_state: torch.Tensor,  # [batch, hidden_dim]
        context: torch.Tensor,  # [batch, hidden_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate agency and readiness to act.
        
        Args:
            internal_state: Current internal state
            context: Current context
            
        Returns:
            Dictionary with agency metrics
        """
        # Encode internal state
        encoded_state = self.state_encoder(internal_state)
        
        # Compute readiness
        readiness = self.readiness_network(encoded_state)
        
        # Compute drives
        drives = self.drive_network(encoded_state)
        
        # Compute autonomy
        autonomy = self.autonomy_network(context)
        
        # Compute action initiation probability
        # Create drive embedding from overall drive
        drive_embedding = encoded_state * drives['overall'].unsqueeze(-1)
        initiation_prob = self.initiation_network(drive_embedding, context)
        
        return {
            'activation_readiness': readiness,
            'drives': drives,
            'autonomy': autonomy,
            'initiation_probability': initiation_prob,
            'agency_embedding': encoded_state,
        }


class WillNetwork(nn.Module):
    """Models the 'will' or volition to act."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Will strength computation
        self.will_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Will direction (what the will is directed toward)
        self.direction_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(
        self,
        internal_drive: torch.Tensor,
        external_context: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute will strength and direction.
        
        Args:
            internal_drive: Internal drive state
            external_context: External context
            
        Returns:
            Dictionary with will metrics
        """
        combined = torch.cat([internal_drive, external_context], dim=-1)
        
        will_strength = self.will_net(combined).squeeze(-1)
        will_direction = self.direction_net(internal_drive)
        
        return {
            'will_strength': will_strength,
            'will_direction': will_direction,
        }


def create_agency_module(config) -> AgencySimulationModule:
    """Factory function to create agency simulation module from config."""
    return AgencySimulationModule(
        hidden_dim=config.model_scale.hidden_dim
    )
