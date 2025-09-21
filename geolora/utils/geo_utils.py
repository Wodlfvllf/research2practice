# utils/geo_utils.py
import torch
import numpy as np
from typing import Tuple, Optional

def economic_qr(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute economic QR decomposition
    
    Args:
        A: Input matrix (m x n)
    
    Returns:
        Q: Orthonormal matrix (m x min(m,n))
        R: Upper triangular matrix (min(m,n) x n)
    """
    Q, R = torch.qr(A)
    return Q, R

def truncated_svd(A: torch.Tensor, k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute truncated SVD
    
    Args:
        A: Input matrix
        k: Number of components to keep (if None, keep all)
    
    Returns:
        U, S, V: SVD components where A ≈ U @ diag(S) @ V^T
    """
    U, S, V = torch.svd(A)
    
    if k is not None:
        k = min(k, len(S))
        U = U[:, :k]
        S = S[:k]
        V = V[:, :k]
    
    return U, S, V


def compute_residuals(X: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    Compute residuals orthogonal to given basis
    
    Args:
        X: Input matrix
        basis: Orthonormal basis matrix
    
    Returns:
        Residuals orthogonal to basis
    """
    coeffs = basis.T @ X
    projection = basis @ coeffs
    residuals = X - projection
    return residuals

def check_orthogonality(Q: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check if matrix has orthonormal columns"""
    QtQ = Q.T @ Q
    I = torch.eye(Q.shape[1], device=Q.device, dtype=Q.dtype)
    return torch.allclose(QtQ, I, atol=tol)

def reorthogonalize(Q: torch.Tensor) -> torch.Tensor:
    """Reorthogonalize matrix using QR decomposition"""
    Q_new, _ = torch.qr(Q)
    return Q_new
