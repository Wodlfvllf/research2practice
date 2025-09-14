"""
Reconstruction Loss (L_recon) Implementation
Simple pixel-wise MSE loss between original and reconstructed images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """
    Implements pixel-wise Mean Squared Error (MSE) reconstruction loss.
    
    This is the most straightforward component of the total loss, ensuring
    numerical accuracy between input and reconstructed images.
    """
    
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                           'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, reconstructed, original, mask):
        """
        Calculate MSE loss between reconstructed and original images.
        
        Args:
            reconstructed (torch.Tensor): Reconstructed images from autoencoder
                                        Shape: (B, C, H, W)
            original (torch.Tensor): Original input images
                                   Shape: (B, C, H, W)
        
        Returns:
            torch.Tensor: MSE reconstruction loss
        """
        return F.mse_loss(reconstructed, original, reduction=self.reduction, mask=mask)

class L1ReconstructionLoss(nn.Module):
    """
    Alternative reconstruction loss using L1 (MAE) instead of MSE.
    Sometimes provides better gradient properties.
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, reconstructed, original, mask):
        """
        Calculate L1 loss between reconstructed and original images.
        
        Args:
            reconstructed (torch.Tensor): Reconstructed images from autoencoder
            original (torch.Tensor): Original input images
        
        Returns:
            torch.Tensor: L1 reconstruction loss
        """
        return F.l1_loss(reconstructed, original, reduction=self.reduction, mask=mask)
    


class CombinedReconstructionLoss(nn.Module):
    """
    Combined L1 + MSE reconstruction loss for better training stability.
    """
    
    def __init__(self, l1_weight=1.0, mse_weight=1.0, reduction='mean'):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.reduction = reduction
    
    def forward(self, reconstructed, original, mask):
        """
        Calculate combined L1 + MSE reconstruction loss.
        
        Args:
            reconstructed (torch.Tensor): Reconstructed images from autoencoder
            original (torch.Tensor): Original input images
        
        Returns:
            torch.Tensor: Combined reconstruction loss
        """
        l1_loss = F.l1_loss(reconstructed, original, reduction=self.reduction, mask = mask)
        mse_loss = F.mse_loss(reconstructed, original, reduction=self.reduction, mask = mask)
        
        return self.l1_weight * l1_loss + self.mse_weight * mse_loss


