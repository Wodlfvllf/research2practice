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

