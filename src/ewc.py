"""
Elastic Weight Consolidation (EWC) implementation for continual learning.

This module provides EWC as a baseline comparison method for MEOs.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np


class EWC(nn.Module):
    """Elastic Weight Consolidation implementation."""
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 100.0):
        """
        Initialize EWC.
        
        Args:
            model: Neural network model
            lambda_ewc: EWC regularization strength
        """
        super().__init__()
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_info = {}
        self.optimal_params = {}
        
    def compute_fisher_info(self, dataloader, num_samples: int = 1000):
        """
        Compute Fisher Information Matrix (diagonal approximation).
        
        Args:
            dataloader: DataLoader for computing Fisher information
            num_samples: Number of samples to use for estimation
        """
        self.model.eval()
        fisher_info = {}
        
        # Initialize Fisher info
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
                
        # Compute Fisher information
        sample_count = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
                
            data, target = data.cuda() if torch.cuda.is_available() else data, target
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            # Backward pass to get gradients
            loss.backward()
            
            # Accumulate squared gradients (Fisher information)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
                    
            sample_count += data.size(0)
            
        # Average Fisher information
        for name in fisher_info:
            fisher_info[name] /= sample_count
            
        self.fisher_info = fisher_info
        
    def save_optimal_params(self):
        """Save current model parameters as optimal for EWC penalty."""
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
                
    def compute_ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.
        
        Returns:
            EWC penalty term
        """
        if not self.fisher_info or not self.optimal_params:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
            
        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_info:
                fisher = self.fisher_info[name]
                optimal = self.optimal_params[name]
                ewc_loss += torch.sum(fisher * (param - optimal) ** 2)
                
        return 0.5 * self.lambda_ewc * ewc_loss


# Example usage
if __name__ == "__main__":
    # Simple example with a small model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    ewc = EWC(model, lambda_ewc=100.0)
    
    # Simulate some data
    dummy_data = torch.randn(100, 784)
    dummy_targets = torch.randint(0, 10, (100,))
    
    # Create a simple dataloader
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(dummy_data, dummy_targets)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # Compute Fisher information
    ewc.compute_fisher_info(dataloader, num_samples=100)
    
    # Save optimal parameters
    ewc.save_optimal_params()
    
    # Compute EWC loss
    ewc_loss = ewc.compute_ewc_loss()
    print(f"EWC loss: {ewc_loss.item():.4f}")
