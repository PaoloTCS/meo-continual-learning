# MEO: Mask Evolution Operators for Continual Learning Stability

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

This repository contains the implementation of **Mask Evolution Operators (MEOs)**, a lightweight activation-level mechanism that applies adaptive masks as a restoring force to stabilize internal representations during continual learning.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/PaoloTCS/meo-continual-learning.git
cd meo-continual-learning

# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py --config configs/meo_cifar100.yaml
```

## 📖 Overview

MEOs provide a complementary **activation-space** mechanism to weight-based methods for continual learning. They directly damp harmful representation drift while allowing principled evolution through:

- **Adaptive activation masks** that exert restoring forces toward references
- **Evolution operators** (identity/EMA/subspace) for controlled plasticity
- **Open- vs. closed-loop** timing strategies
- **Drift metrics** for stability monitoring

## 🏗️ Repository Structure

```
meo-continual-learning/
├─ paper/                    # Research paper and assets
│  ├─ MEO_Paper_v2_clean.tex
│  ├─ MEO_Paper_v2_clean.pdf
│  ├─ drift_placeholder.png
│  └─ alpha_sweep_placeholder.png
├─ src/                      # Source code
│  ├─ meo.py                # Mask evolution operators
│  ├─ ewc.py                # EWC implementation
│  ├─ data.py               # CIFAR-100 data loading
│  └─ train.py              # Training entrypoints
├─ configs/                  # Configuration files
│  ├─ meo_cifar100.yaml
│  └─ ewc_cifar100.yaml
├─ results/                  # Outputs (gitignored)
│  ├─ logs/                 # Training logs
│  └─ figures/              # Generated plots
└─ docs/                     # Documentation
```

## 🔬 Key Results

On a 10-task split of CIFAR-100 with ResNet-50:

| Method | Final Accuracy | Drift Metric |
|--------|----------------|--------------|
| Finetune | 51.2% | High |
| EWC (tuned) | 62.0% | Medium |
| **MEO (identity)** | **69.1%** | **Near-zero** |

## 📚 Paper

The research paper is available in the `paper/` directory:
- **LaTeX source**: `MEO_Paper_v2_clean.tex`
- **Compiled PDF**: `MEO_Paper_v2_clean.pdf`

## 🛠️ Implementation

### Core Components

- **`meo.py`**: Implementation of mask evolution operators
- **`ewc.py`**: Elastic Weight Consolidation baseline
- **`data.py`**: CIFAR-100 data loading and task splitting
- **`train.py`**: Training loops and evaluation

### Configuration

YAML configuration files control hyperparameters:
- Learning rates and schedules
- MEO stiffness parameters (α)
- Evolution operator selection
- Training protocols

## 📊 Usage Examples

```python
from src.meo import MEO
from src.data import CIFAR100Continual

# Initialize MEO with identity evolution
meo = MEO(evolution_type='identity', alpha=0.1)

# Train on continual learning tasks
for task_id in range(10):
    task_data = continual_data.get_task(task_id)
    meo.train_on_task(task_data)
```

## 🔧 Dependencies

- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy
- matplotlib
- pyyaml

## 📄 License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Paper**: CC BY 4.0

## 🤝 Citation

If you use this work, please cite:

```bibtex
@software{meo_continual_learning,
  title={MEO: Mask Evolution Operators for Continual Learning Stability},
  author={Pignatelli di Montecalvo, Paolo},
  year={2025},
  url={https://github.com/PaoloTCS/meo-continual-learning}
}
```

## 📞 Contact

- **Author**: Paolo Pignatelli di Montecalvo
- **Email**: [Your Email]
- **Affiliation**: Independent Researcher; Verbum Technologies

## 🙏 Acknowledgments

Thanks to the continual learning research community and contributors to this project.

---

**Note**: This is a research implementation. For production use, additional testing and validation is recommended.
