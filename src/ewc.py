"""
Elastic Weight Consolidation (EWC) implementation for continual learning.

This module provides EWC as a baseline comparison method for MEOs.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np


class EWC:
    """Elastic Weight Consolidation implementation."""
    
    def __init__(self, lambda_: float = 100.0, gamma: float = 0.9, mode: str = "online"):
        """
        Initialize EWC.
        
        Args:
            lambda_: EWC regularization strength
            gamma: Fisher update momentum (for online mode)
            mode: EWC mode ('online' or 'offline')
        """
        self.lambda_ = lambda_
        self.gamma = gamma
        self.mode = mode
        self.fisher_info = {}
        self.optimal_params = {}
        self.consolidated_fisher = {}
        
    def update_fisher(self, model: nn.Module, dataloader, device: torch.device, batches: int = 200):
        """
        Update Fisher Information Matrix (diagonal approximation).
        
        Args:
            model: Neural network model
            dataloader: DataLoader for computing Fisher information
            device: Device to use for computation
            batches: Number of batches to use for estimation
        """
        model.eval()
        fisher_info = {}
        
        # Initialize Fisher info
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
                
        # Compute Fisher information
        batch_count = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_count >= batches:
                break
                
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            # Backward pass to get gradients
            loss.backward()
            
            # Accumulate squared gradients (Fisher information)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
                    
            batch_count += 1
            
        # Average Fisher information
        for name in fisher_info:
            fisher_info[name] /= max(batch_count, 1)
            
        # Update consolidated Fisher (online mode)
        if self.mode == "online":
            if not self.consolidated_fisher:
                self.consolidated_fisher = fisher_info
            else:
                for name in fisher_info:
                    if name in self.consolidated_fisher:
                        self.consolidated_fisher[name] = (
                            self.gamma * self.consolidated_fisher[name] + 
                            (1 - self.gamma) * fisher_info[name]
                        )
                    else:
                        self.consolidated_fisher[name] = fisher_info[name]
        else:
            # Offline mode: replace
            self.consolidated_fisher = fisher_info
        
    def save_optimal_params(self, model: nn.Module):
        """Save current model parameters as optimal for EWC penalty."""
        self.optimal_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
                
    def compute_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC regularization loss.
        
        Args:
            model: Neural network model
            
        Returns:
            EWC penalty term
        """
        if not self.consolidated_fisher or not self.optimal_params:
            return torch.tensor(0.0, device=next(model.parameters()).device)
            
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.consolidated_fisher:
                fisher = self.consolidated_fisher[name]
                optimal = self.optimal_params[name]
                ewc_loss += torch.sum(fisher * (param - optimal) ** 2)
                
        return 0.5 * self.lambda_ * ewc_loss


# Example usage
if __name__ == "__main__":
    # Simple example with a small model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    ewc = EWC(lambda_=100.0, gamma=0.9, mode="online")
    
    # Simulate some data
    dummy_data = torch.randn(100, 784)
    dummy_targets = torch.randint(0, 10, (100,))
    
    # Create a simple dataloader
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(dummy_data, dummy_targets)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # Update Fisher information
    device = torch.device('cpu')
    ewc.update_fisher(model, dataloader, device, batches=3)
    
    # Save optimal parameters
    ewc.save_optimal_params(model)
    
    # Compute EWC loss
    ewc_loss = ewc.compute_ewc_loss(model)
    print(f"EWC loss: {ewc_loss.item():.4f}")
