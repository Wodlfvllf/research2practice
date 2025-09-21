# config/geolora_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class GeoLoRAConfig:
    # Basic LoRA parameters
    rank_init: int = 8  # Initial rank r
    alpha: float = 1e-4  # Initialization scale for S
    
    # Learning parameters
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    
    # GeoLoRA specific parameters
    tau: float = 1e-3  # Truncation threshold factor
    epsilon_resid: float = 1e-6  # Residual column norm cutoff
    max_new_cols: Optional[int] = None  # Max new columns (default: same as rank)
    
    # Optimizer settings
    use_small_matrix_optimizer: bool = True
    optimizer_type: str = "adam"  # "adam" or "sgd"
    
    # Truncation strategy
    truncation_strategy: str = "local_threshold"  # "local_threshold", "fixed_rank", "global_budget"
    global_budget: Optional[float] = None  # For global budget strategy
    
    # Numerical stability
    orthogonal_regularization: bool = False
    damping_epsilon: float = 1e-6
