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
            
    def _initialize_factors(self):
        """Initialize U, S, V according to the specification"""
        # Sample random matrices and orthonormalize via QR
        A_U = torch.randn(self.out_features, self.current_rank)
        A_V = torch.randn(self.in_features, self.current_rank)
        
        # Economic QR decomposition
        U_init, _ = torch.qr(A_U)
        V_init, _ = torch.qr(A_V)
        
        # Store as parameters
        self.U = nn.Parameter(U_init[:, :self.current_rank].clone())  # n x r
        self.V = nn.Parameter(V_init[:, :self.current_rank].clone())  # m x r
        
        # Initialize S small (diagonal stored as vector)
        self.s = nn.Parameter(torch.full((self.current_rank,), 
                                       self.config.alpha, dtype=torch.float32))
        
        # Sanity check
        assert torch.allclose(self.U.T @ self.U, torch.eye(self.current_rank), atol=1e-5)
        assert torch.allclose(self.V.T @ self.V, torch.eye(self.current_rank), atol=1e-5)
    
    def _update_cache(self):
        """Update cached coefficient matrices K and L"""
        S_diag = torch.diag(self.s)
        self.K_cache = self.U @ S_diag  # n x r
        self.L_cache = self.V @ S_diag.T  # m x r
        
        def _setup_small_optimizer(self):
        """Setup optimizer for small matrices"""
        small_params = [self.s]  # Only optimize S in coefficient updates
        
        if self.config.optimizer_type == "adam":
            self.small_optimizer = torch.optim.Adam(
                small_params, 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            self.small_optimizer = torch.optim.SGD(
                small_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: y = base_layer(x) + U @ S @ V^T @ x"""
        # Base layer output
        if self.base_layer is not None:
            base_out = self.base_layer(x)
        else:
            base_out = 0
        
        # LoRA adaptation: W @ x = U @ S @ V^T @ x
        if self.current_rank > 0:
            # Efficient computation: (U @ S) @ (V^T @ x)
            v_x = self.V.T @ x.T  # r x batch_size
            s_v_x = self.s.unsqueeze(1) * v_x  # r x batch_size (broadcast)
            adaptation = self.U @ s_v_x  # n x batch_size
            adaptation = adaptation.T  # batch_size x n
        else:
            adaptation = 0
        
        return base_out + adaptation
    
    def get_full_weight(self) -> torch.Tensor:
        """Reconstruct full weight matrix W = U @ S @ V^T"""
        if self.current_rank == 0:
            return torch.zeros(self.out_features, self.in_features)
        
        S_diag = torch.diag(self.s)
        return self.U @ S_diag @ self.V.T
    
    def geolora_step(self, ambient_grad: torch.Tensor, learning_rate: float):
        """
        Perform one GeoLoRA optimization step
        
        Args:
            ambient_grad: Gradient w.r.t. full weight matrix W (n x m)
            learning_rate: Step size λ
        """
        if self.current_rank == 0:
            return  # Nothing to update
        
        # Step 3: Coefficient updates
        K_new, L_new, s_new = self._coefficient_updates(ambient_grad, learning_rate)
        
        # Step 4-5: Compute residuals and new basis directions
        R_U, R_V = self._compute_residuals(K_new, L_new)
        U_tilde, V_tilde, k_u, k_v = self._orthonormalize_residuals(R_U, R_V)
        
        if k_u == 0 and k_v == 0:
            # No new directions, just update coefficients
            self.s.data = s_new
            self._update_cache()
            return
        
        # Step 6: Assemble augmented coefficient matrix
        S_hat = self._assemble_augmented_matrix(s_new, K_new, L_new, U_tilde, V_tilde)
        
        # Step 8: SVD and truncation
        U_new, s_new_truncated, V_new, new_rank = self._svd_and_truncate(
            S_hat, U_tilde, V_tilde, k_u, k_v
        )
        
        # Step 9: Update layer parameters
        self._update_parameters(U_new, s_new_truncated, V_new, new_rank)
    
    def _coefficient_updates(self, G: torch.Tensor, lr: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step 3: Update coefficient matrices K, L, S"""
        # Current matrices
        U, V = self.U, self.V
        s_old = self.s
        
        # Compute updates (discrete Euler)
        # S update: s_new = s - λ * diag(U^T @ G @ V)
        UGV = U.T @ G @ V  # r x r
        s_new = s_old - lr * torch.diag(UGV)
        
        # K update: K_new = K - λ * (G @ V)
        K_new = self.K_cache - lr * (G @ V)  # n x r
        
        # L update: L_new = L - λ * (G^T @ U)
        L_new = self.L_cache - lr * (G.T @ U)  # m x r
        
        return K_new, L_new, s_new