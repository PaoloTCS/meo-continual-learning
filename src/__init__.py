"""
MEO: Mask Evolution Operators for Continual Learning Stability

This package provides implementations of Mask Evolution Operators (MEOs),
a lightweight activation-level mechanism for continual learning stability.

Main components:
- meo: Core MEO implementation with evolution operators
- ewc: Elastic Weight Consolidation baseline
- data: CIFAR-100 continual learning data management
- train: Training and evaluation scripts
"""

__version__ = "1.0.0"
__author__ = "Paolo Pignatelli di Montecalvo"
__email__ = "paolo.pignatelli@verbumtechnologies.com"

from .meo import MEO, MaskEvolutionOperator, IdentityOperator, EMAOperator
from .ewc import EWC
from .data import CIFAR100Continual
from .train import ContinualTrainer

__all__ = [
    "MEO",
    "MaskEvolutionOperator", 
    "IdentityOperator",
    "EMAOperator",
    "EWC",
    "CIFAR100Continual",
    "ContinualTrainer"
]
