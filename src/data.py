"""
Data loading and task splitting for CIFAR-100 continual learning experiments.

This module handles:
- CIFAR-100 dataset loading
- 10-task class-incremental splitting
- Data augmentation and preprocessing
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from typing import List, Tuple, Dict
import os


class CIFAR100Continual:
    """CIFAR-100 continual learning dataset manager."""
    
    def __init__(self, 
                 root: str = './data',
                 num_tasks: int = 10,
                 batch_size: int = 128,
                 num_workers: int = 4,
                 seed: int = 42):
        """
        Initialize CIFAR-100 continual learning setup.
        
        Args:
            root: Data directory
            num_tasks: Number of tasks to split into
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            seed: Random seed for reproducibility
        """
        self.root = root
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # CIFAR-100 has 100 classes
        self.num_classes = 100
        self.classes_per_task = self.num_classes // num_tasks
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761))
        ])
        
        # Load datasets
        self._load_datasets()
        self._create_task_splits()
        
    def _load_datasets(self):
        """Load CIFAR-100 train and test datasets."""
        # Download if not exists
        self.train_dataset = torchvision.datasets.CIFAR100(
            root=self.root, 
            train=True, 
            download=True,
            transform=self.train_transform
        )
        
        self.test_dataset = torchvision.datasets.CIFAR100(
            root=self.root, 
            train=False, 
            download=True,
            transform=self.test_transform
        )
        
        print(f"Loaded CIFAR-100: {len(self.train_dataset)} train, {len(self.test_dataset)} test")
        
    def _create_task_splits(self):
        """Create task-specific class splits."""
        # Get all class indices
        all_classes = list(range(self.num_classes))
        
        # Shuffle classes for random task assignment
        np.random.shuffle(all_classes)
        
        # Create task splits
        self.task_classes = {}
        for task_id in range(self.num_tasks):
            start_idx = task_id * self.classes_per_task
            end_idx = start_idx + self.classes_per_task
            self.task_classes[task_id] = all_classes[start_idx:end_idx]
            
        print(f"Created {self.num_tasks} tasks with {self.classes_per_task} classes each")
        
    def get_task_data(self, task_id: int, split: str = 'train') -> DataLoader:
        """
        Get data for a specific task.
        
        Args:
            task_id: Task identifier (0 to num_tasks-1)
            split: Data split ('train' or 'test')
            
        Returns:
            DataLoader for the specified task
        """
        if task_id not in self.task_classes:
            raise ValueError(f"Invalid task_id: {task_id}")
            
        if split not in ['train', 'test']:
            raise ValueError(f"Invalid split: {split}")
            
        # Get class indices for this task
        task_class_indices = self.task_classes[task_id]
        
        # Filter dataset by classes
        if split == 'train':
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset
            
        # Create task-specific dataset
        task_indices = []
        for idx, (_, label) in enumerate(dataset):
            if label in task_class_indices:
                task_indices.append(idx)
                
        task_dataset = Subset(dataset, task_indices)
        
        # Create DataLoader
        dataloader = DataLoader(
            task_dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return dataloader
        
    def get_all_tasks_data(self, split: str = 'train') -> Dict[int, DataLoader]:
        """
        Get data for all tasks.
        
        Args:
            split: Data split ('train' or 'test')
            
        Returns:
            Dictionary mapping task_id to DataLoader
        """
        return {task_id: self.get_task_data(task_id, split) 
                for task_id in range(self.num_tasks)}
        
    def get_task_info(self, task_id: int) -> Dict:
        """
        Get information about a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dictionary with task information
        """
        if task_id not in self.task_classes:
            raise ValueError(f"Invalid task_id: {task_id}")
            
        # Get class names for this task
        class_names = [self.train_dataset.classes[class_idx] 
                      for class_idx in self.task_classes[task_id]]
        
        return {
            'task_id': task_id,
            'classes': self.task_classes[task_id],
            'class_names': class_names,
            'num_classes': len(self.task_classes[task_id])
        }
        
    def get_accumulated_data(self, up_to_task: int, split: str = 'train') -> DataLoader:
        """
        Get accumulated data from task 0 up to specified task.
        
        Args:
            up_to_task: Last task to include (inclusive)
            split: Data split ('train' or 'test')
            
        Returns:
            DataLoader with accumulated data
        """
        if up_to_task < 0 or up_to_task >= self.num_tasks:
            raise ValueError(f"Invalid up_to_task: {up_to_task}")
            
        # Collect indices from all tasks up to up_to_task
        all_indices = []
        for task_id in range(up_to_task + 1):
            task_class_indices = self.task_classes[task_id]
            
            if split == 'train':
                dataset = self.train_dataset
            else:
                dataset = self.test_dataset
                
            for idx, (_, label) in enumerate(dataset):
                if label in task_class_indices:
                    all_indices.append(idx)
                    
        # Create accumulated dataset
        if split == 'train':
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset
            
        accumulated_dataset = Subset(dataset, all_indices)
        
        # Create DataLoader
        dataloader = DataLoader(
            accumulated_dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return dataloader


# Example usage
if __name__ == "__main__":
    # Initialize continual learning dataset
    continual_data = CIFAR100Continual(
        root='./data',
        num_tasks=10,
        batch_size=128
    )
    
    # Get data for first task
    task_0_train = continual_data.get_task_data(0, 'train')
    task_0_test = continual_data.get_task_data(0, 'test')
    
    print(f"Task 0 train batches: {len(task_0_train)}")
    print(f"Task 0 test batches: {len(task_0_test)}")
    
    # Get task information
    task_info = continual_data.get_task_info(0)
    print(f"Task 0 classes: {task_info['class_names']}")
    
    # Get accumulated data up to task 2
    accumulated_train = continual_data.get_accumulated_data(2, 'train')
    print(f"Accumulated data (tasks 0-2) train batches: {len(accumulated_train)}")
