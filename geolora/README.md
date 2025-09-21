# README.md
# GeoLoRA: Geometric Low-Rank Adaptation

A PyTorch implementation of Geometric Low-Rank Adaptation (GeoLoRA), an advanced parameter-efficient fine-tuning method that dynamically adapts the rank of LoRA layers using geometric optimization on the Grassmann manifold.

## Features

- **Dynamic Rank Adaptation**: Automatically adjusts the rank of adaptation layers during training
- **Geometric Optimization**: Uses manifold optimization to maintain orthonormal bases
- **Parameter Efficiency**: Significantly reduces the number of trainable parameters
- **Flexible Configuration**: Extensive hyperparameter control for different use cases
- **Multiple Truncation Strategies**: Local threshold, fixed rank, and global budget approaches

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/geolora.git
cd geolora

# Install the package
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from geolora.config import GeoLoRAConfig
from geolora.models import GeoLoRAModel

# Create a base model
base_model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Configure GeoLoRA
config = GeoLoRAConfig(
    rank_init=8,           # Initial rank
    learning_rate=1e-3,    # Learning rate
    tau=1e-3,             # Truncation threshold
    alpha=1e-4            # Initialization scale
)

# Wrap with GeoLoRA
model = GeoLoRAModel(base_model, config, target_modules=["Linear"])
```

### Training

```python
from geolora.training import GeoLoRATrainer
from geolora.data import get_dataloaders

# Load data
train_loader, val_loader = get_dataloaders("mnist", batch_size=128)

# Create trainer
trainer = GeoLoRATrainer(model, config)

# Train
history = trainer.train(train_loader, val_loader, num_epochs=10)
```

### Command Line Interface

```bash
# Train on MNIST
python geolora_main.py --dataset mnist --epochs 10 --rank 8 --lr 1e-3

# Train on CIFAR-10 with custom parameters
python geolora_main.py --dataset cifar10 --epochs 20 --rank 16 --tau 1e-2 --alpha 1e-3
```

## Algorithm Overview

GeoLoRA implements a sophisticated optimization procedure that:

1. **Maintains Factorization**: Keeps weights in the form W = U @ S @ V^T with orthonormal U, V
2. **Updates Coefficients**: Uses stable gradient updates on coefficient matrices K, L, S
3. **Computes Residuals**: Finds new gradient directions orthogonal to current bases
4. **Augments Subspace**: Expands the representation space with new directions
5. **Optimal Truncation**: Uses SVD to find the best low-rank approximation
6. **Rank Adaptation**: Dynamically adjusts rank based on truncation criteria

## Configuration Options

### Core Parameters

- `rank_init`: Initial rank of adaptation layers (default: 8)
- `learning_rate`: Learning rate for optimization (default: 1e-3)
- `tau`: Truncation threshold factor (default: 1e-3)
- `alpha`: Initialization scale for S matrix (default: 1e-4)

### Advanced Options

- `truncation_strategy`: "local_threshold", "fixed_rank", or "global_budget"
- `epsilon_resid`: Residual column norm cutoff (default: 1e-6)
- `use_small_matrix_optimizer`: Enable Adam/SGD for small matrices
- `orthogonal_regularization`: Add orthogonality constraints

## Project Structure

```
geolora/
├── config/
│   └── geolora_config.py     # Configuration classes
├── models/
│   ├── geolora_layer.py      # Core GeoLoRA layer implementation
│   └── geolora_model.py      # Model wrapper
├── training/
│   └── geolora_trainer.py    # Training loop and optimization
├── utils/
│   └── geo_utils.py          # Geometric utilities (QR, SVD, etc.)
├── data/
│   └── dataset.py            # Dataset loading utilities
└── geolora_main.py           # Main training script
```

## Mathematical Background

GeoLoRA is based on optimization on the Grassmann manifold, where the adaptation weights are factorized as:

W = U S V^T

Where:
- U ∈ R^(n×r): Left orthonormal basis
- S ∈ R^(r×r): Diagonal scaling matrix  
- V ∈ R^(m×r): Right orthonormal basis

The algorithm maintains these constraints while allowing the rank r to adapt dynamically based on the geometry of the loss landscape.
```