"""
Mask Evolution Operators (MEOs) for Continual Learning Stability.

This module implements the core MEO functionality including:
- Activation-level mask computation
- Evolution operators (identity, EMA, subspace)
- Open- vs. closed-loop timing strategies
- Drift metrics for stability monitoring
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class MaskEvolutionOperator:
    """Base class for mask evolution operators."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize the evolution operator.
        
        Args:
            alpha: Stiffness parameter controlling mask strength
        """
        self.alpha = alpha
        self.references = {}
        
    def update_reference(self, layer_name: str, activations: torch.Tensor):
        """Update reference activations for a layer."""
        raise NotImplementedError
        
    def compute_mask(self, layer_name: str, activations: torch.Tensor) -> torch.Tensor:
        """Compute correction mask for given activations."""
        raise NotImplementedError


class IdentityOperator(MaskEvolutionOperator):
    """Identity evolution operator - maximal rigidity stress test."""
    
    def update_reference(self, layer_name: str, activations: torch.Tensor):
        """Keep reference unchanged (identity operation)."""
        if layer_name not in self.references:
            self.references[layer_name] = activations.detach().clone()
        # Identity: do nothing to reference
        
    def compute_mask(self, layer_name: str, activations: torch.Tensor) -> torch.Tensor:
        """Compute mask based on deviation from fixed reference."""
        if layer_name not in self.references:
            return torch.zeros_like(activations)
            
        error = activations - self.references[layer_name]
        # Normalize per-channel for stability
        if error.dim() > 1:
            std = error.std(dim=0, keepdim=True)
            error = error / (std + 1e-8)
            
        return self.alpha * error


class EMAOperator(MaskEvolutionOperator):
    """Exponential Moving Average evolution operator."""
    
    def __init__(self, alpha: float = 0.1, eta: float = 0.02):
        """
        Initialize EMA operator.
        
        Args:
            alpha: Stiffness parameter
            eta: EMA momentum parameter
        """
        super().__init__(alpha)
        self.eta = eta
        
    def update_reference(self, layer_name: str, activations: torch.Tensor):
        """Update reference using EMA."""
        batch_mean = activations.mean(dim=0, keepdim=True)
        
        if layer_name not in self.references:
            self.references[layer_name] = batch_mean.detach().clone()
        else:
            self.references[layer_name] = (
                (1 - self.eta) * self.references[layer_name] + 
                self.eta * batch_mean
            )
            
    def compute_mask(self, layer_name: str, activations: torch.Tensor) -> torch.Tensor:
        """Compute mask based on deviation from evolving reference."""
        if layer_name not in self.references:
            return torch.zeros_like(activations)
            
        error = activations - self.references[layer_name]
        # Normalize per-channel for stability
        if error.dim() > 1:
            std = error.std(dim=0, keepdim=True)
            error = error / (std + 1e-8)
            
        return self.alpha * error


class MEO(nn.Module):
    """Main MEO class implementing the mask evolution framework."""
    
    def __init__(self, 
                 evolution_type: str = 'identity',
                 alpha: float = 0.1,
                 eta: float = 0.02,
                 timing: str = 'open_loop'):
        """
        Initialize MEO.
        
        Args:
            evolution_type: Type of evolution operator ('identity', 'ema', 'subspace')
            alpha: Stiffness parameter
            eta: EMA momentum (for EMA operator)
            timing: Timing strategy ('open_loop' or 'closed_loop')
        """
        super().__init__()
        self.timing = timing
        
        if evolution_type == 'identity':
            self.evolution_op = IdentityOperator(alpha)
        elif evolution_type == 'ema':
            self.evolution_op = EMAOperator(alpha, eta)
        else:
            raise ValueError(f"Unknown evolution type: {evolution_type}")
            
        self.masks = {}
        
    def forward(self, layer_name: str, activations: torch.Tensor) -> torch.Tensor:
        """
        Apply MEO correction to activations.
        
        Args:
            layer_name: Name of the layer
            activations: Input activations
            
        Returns:
            Corrected activations
        """
        if self.timing == 'open_loop':
            # Apply previous mask, then compute new one
            if layer_name in self.masks:
                corrected = activations - self.masks[layer_name]
            else:
                corrected = activations
                
            # Compute new mask from uncorrected activations
            self.masks[layer_name] = self.evolution_op.compute_mask(
                layer_name, activations
            )
            
        else:  # closed_loop
            # Compute mask and apply immediately
            mask = self.evolution_op.compute_mask(layer_name, activations)
            corrected = activations - mask
            self.masks[layer_name] = mask
            
        return corrected
        
    def update_reference(self, layer_name: str, activations: torch.Tensor):
        """Update reference activations for a layer."""
        self.evolution_op.update_reference(layer_name, activations)
        
    def get_drift_metric(self, layer_names: List[str]) -> float:
        """
        Compute drift metric across layers.
        
        Args:
            layer_names: List of layer names to include
            
        Returns:
            Average normalized L2 drift
        """
        total_drift = 0.0
        valid_layers = 0
        
        for layer_name in layer_names:
            if layer_name in self.evolution_op.references:
                ref = self.evolution_op.references[layer_name]
                if layer_name in self.masks:
                    current = ref + self.masks[layer_name] / self.evolution_op.alpha
                    
                    # Compute normalized L2 distance
                    ref_mean = ref.mean(dim=0)
                    current_mean = current.mean(dim=0)
                    
                    numerator = torch.norm(current_mean - ref_mean, p=2)
                    denominator = torch.norm(ref_mean, p=2) + 1e-8
                    
                    drift = numerator / denominator
                    total_drift += drift.item()
                    valid_layers += 1
                    
        return total_drift / max(valid_layers, 1)


# Example usage
if __name__ == "__main__":
    # Initialize MEO with identity evolution
    meo = MEO(evolution_type='identity', alpha=0.1)
    
    # Simulate activations for a layer
    layer_name = "layer1"
    activations = torch.randn(32, 64)  # batch_size=32, features=64
    
    # Update reference (first call)
    meo.update_reference(layer_name, activations)
    
    # Apply correction
    corrected = meo(layer_name, activations)
    
    print(f"Original activations shape: {activations.shape}")
    print(f"Corrected activations shape: {corrected.shape}")
    print(f"Drift metric: {meo.get_drift_metric([layer_name]):.4f}")
