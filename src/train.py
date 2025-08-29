"""
Training script for MEO continual learning experiments.

This module provides training entrypoints for:
- MEO training with different evolution operators
- EWC baseline training
- Evaluation and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import yaml
import argparse
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from meo import MEO
from ewc import EWC
from data import CIFAR100Continual


class ContinualTrainer:
    """Main training class for continual learning experiments."""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seed
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
        # Initialize data
        self.data_manager = CIFAR100Continual(
            root=self.config['data']['root'],
            num_tasks=self.config['data']['num_tasks'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            seed=self.config['seed']
        )
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize method (MEO or EWC)
        self.method = self.config['method']['type']
        if self.method == 'meo':
            self.meo = MEO(
                evolution_type=self.config['method']['evolution_type'],
                alpha=self.config['method']['alpha'],
                eta=self.config['method'].get('eta', 0.02),
                timing=self.config['method'].get('timing', 'open_loop')
            )
            self.ewc = None
        elif self.method == 'ewc':
            self.ewc = EWC(self.model, self.config['method']['lambda_ewc'])
            self.meo = None
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        # Initialize optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            momentum=self.config['training']['momentum'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs_per_task']
        )
        
        # Results storage
        self.results = {
            'task_accuracies': [],
            'drift_metrics': [],
            'training_losses': [],
            'learning_rates': []
        }
        
    def _create_model(self) -> nn.Module:
        """Create the neural network model."""
        model_name = self.config['model']['name']
        
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=False)
            # Modify final layer for CIFAR-100
            model.fc = nn.Linear(model.fc.in_features, 100)
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 100)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return model
        
    def train_task(self, task_id: int) -> Dict:
        """
        Train on a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dictionary with training results
        """
        print(f"\n=== Training on Task {task_id} ===")
        
        # Get task data
        train_loader = self.data_manager.get_task_data(task_id, 'train')
        test_loader = self.data_manager.get_task_data(task_id, 'test')
        
        # Training loop
        self.model.train()
        task_losses = []
        
        for epoch in range(self.config['training']['epochs_per_task']):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if self.method == 'meo':
                    # Apply MEO corrections during forward pass
                    output = self._forward_with_meo(data)
                else:
                    output = self.model(data)
                    
                # Compute loss
                loss = nn.CrossEntropyLoss()(output, target)
                
                # Add EWC penalty if using EWC
                if self.method == 'ewc':
                    ewc_loss = self.ewc.compute_ewc_loss()
                    loss += ewc_loss
                    
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update MEO references if using MEO
                if self.method == 'meo':
                    self._update_meo_references(data)
                    
            # Update learning rate
            self.scheduler.step()
            
            # Evaluate on test set
            test_acc = self.evaluate_task(task_id, test_loader)
            
            epoch_loss /= num_batches
            task_losses.append(epoch_loss)
            
            print(f"Epoch {epoch+1}/{self.config['training']['epochs_per_task']}: "
                  f"Loss: {epoch_loss:.4f}, Test Acc: {test_acc:.2f}%")
                  
        # Save optimal parameters for EWC
        if self.method == 'ewc':
            self.ewc.save_optimal_params()
            
        # Compute final task accuracy
        final_acc = self.evaluate_task(task_id, test_loader)
        
        # Compute drift metric if using MEO
        drift_metric = 0.0
        if self.method == 'meo':
            drift_metric = self.meo.get_drift_metric(['layer1', 'layer2'])  # Example layers
            
        return {
            'task_id': task_id,
            'final_accuracy': final_acc,
            'drift_metric': drift_metric,
            'training_losses': task_losses
        }
        
    def _forward_with_meo(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass with MEO corrections."""
        # This is a simplified version - in practice, you'd need to hook into
        # the model's forward pass to apply MEO corrections at specific layers
        return self.model(data)
        
    def _update_meo_references(self, data: torch.Tensor):
        """Update MEO reference activations."""
        # This is a simplified version - in practice, you'd need to hook into
        # the model's forward pass to capture activations
        pass
        
    def evaluate_task(self, task_id: int, test_loader) -> float:
        """
        Evaluate model on a specific task.
        
        Args:
            task_id: Task identifier
            test_loader: Test data loader
            
        Returns:
            Test accuracy percentage
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        accuracy = 100 * correct / total
        return accuracy
        
    def evaluate_all_tasks(self) -> List[float]:
        """
        Evaluate model on all tasks seen so far.
        
        Returns:
            List of accuracies for each task
        """
        accuracies = []
        
        for task_id in range(len(self.results['task_accuracies'])):
            test_loader = self.data_manager.get_task_data(task_id, 'test')
            acc = self.evaluate_task(task_id, test_loader)
            accuracies.append(acc)
            
        return accuracies
        
    def run_experiment(self) -> Dict:
        """
        Run the complete continual learning experiment.
        
        Returns:
            Dictionary with experiment results
        """
        print("Starting continual learning experiment...")
        start_time = time.time()
        
        # Train on each task sequentially
        for task_id in range(self.config['data']['num_tasks']):
            # Train on current task
            task_results = self.train_task(task_id)
            
            # Store results
            self.results['task_accuracies'].append(task_results['final_accuracy'])
            self.results['drift_metrics'].append(task_results['drift_metric'])
            self.results['training_losses'].extend(task_results['training_losses'])
            
            # Evaluate on all previous tasks
            all_task_accs = self.evaluate_all_tasks()
            print(f"Task {task_id} completed. All task accuracies: {all_task_accs}")
            
        # Compute final metrics
        final_avg_acc = np.mean(self.results['task_accuracies'])
        final_drift = np.mean(self.results['drift_metrics'])
        
        experiment_time = time.time() - start_time
        
        print(f"\n=== Experiment Complete ===")
        print(f"Final average accuracy: {final_avg_acc:.2f}%")
        print(f"Final average drift: {final_drift:.4f}")
        print(f"Total time: {experiment_time/3600:.2f} hours")
        
        return {
            'final_avg_accuracy': final_avg_acc,
            'final_avg_drift': final_drift,
            'task_accuracies': self.results['task_accuracies'],
            'drift_metrics': self.results['drift_metrics'],
            'experiment_time': experiment_time
        }
        
    def save_results(self, output_dir: str):
        """Save experiment results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results as CSV
        results_df = pd.DataFrame({
            'task_id': range(len(self.results['task_accuracies'])),
            'accuracy': self.results['task_accuracies'],
            'drift_metric': self.results['drift_metrics']
        })
        results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
        
        # Save training losses
        losses_df = pd.DataFrame({
            'epoch': range(len(self.results['training_losses'])),
            'loss': self.results['training_losses']
        })
        losses_df.to_csv(os.path.join(output_dir, 'training_losses.csv'), index=False)
        
        # Create plots
        self._create_plots(output_dir)
        
    def _create_plots(self, output_dir: str):
        """Create and save result plots."""
        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.results['task_accuracies'])), 
                self.results['task_accuracies'], 'bo-')
        plt.xlabel('Task ID')
        plt.ylabel('Accuracy (%)')
        plt.title('Task-wise Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'task_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Drift metric plot
        if any(d != 0 for d in self.results['drift_metrics']):
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(self.results['drift_metrics'])), 
                    self.results['drift_metrics'], 'ro-')
            plt.xlabel('Task ID')
            plt.ylabel('Drift Metric')
            plt.title('Task-wise Drift Metric')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'drift_metric.png'), dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Main training entrypoint."""
    parser = argparse.ArgumentParser(description='MEO Continual Learning Training')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration YAML file')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ContinualTrainer(args.config)
    
    # Run experiment
    results = trainer.run_experiment()
    
    # Save results
    trainer.save_results(args.output_dir)
    
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
