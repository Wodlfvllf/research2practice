# models/geolora_layer.py
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from ..utils.geo_utils import economic_qr, truncated_svd, compute_residuals
from ..config.geolora_config import GeoLoRAConfig

class GeoLoRALayer(nn.Module):
    """
    GeoLoRA factorized layer implementing W = U @ S @ V^T
    
    Maintains orthonormal bases U, V and diagonal matrix S.
    Supports dynamic rank adaptation through geometric optimization.
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 config: GeoLoRAConfig,
                 base_layer: Optional[nn.Module] = None):
        super().__init__()
        
        self.in_features = in_features  # m
        self.out_features = out_features  # n
        self.config = config
        self.current_rank = config.rank_init  # r
        
        # Base layer (frozen)
        self.base_layer = base_layer
        if base_layer:
            for param in base_layer.parameters():
                param.requires_grad = False
        
        # Initialize orthonormal bases U, V and diagonal S
        self._initialize_factors()
        
        # Cache for coefficient matrices
        self.K_cache = None  # U @ S
        self.L_cache = None  # V @ S^T
        self._update_cache()
        
        # Optimizer state for small matrices (if enabled)
        self.small_optimizer = None
        if config.use_small_matrix_optimizer:
            self._setup_small_optimizer()