"""
Reconstruction Loss (L_recon) Implementation
Simple pixel-wise MSE loss between original and reconstructed images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class ReconstructionLoss(nn.Module):
    """
    Implements pixel-wise Mean Squared Error (MSE) reconstruction loss,
    specifically for a Masked Autoencoder (MAE) output.
    
    The loss is calculated ONLY on the patches that were masked.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, reconstruction_patches, original_image, patch_size, mask):
        """
        Calculate MSE loss on the masked patches.
        
        Args:
            reconstruction_patches (torch.Tensor): The output from the MAE decoder.
                                                  Shape: (B, num_patches, patch_features)
            original_image (torch.Tensor): The original input image.
                                           Shape: (B, C, H, W)
            patch_size (int): The size of a single patch (e.g., 4 for a 4x4 patch).
            mask (torch.Tensor): The binary mask from the MAE. 
                                 Shape: (B, num_patches), where 1 indicates a masked patch.
        
        Returns:
            torch.Tensor: MSE reconstruction loss calculated on masked patches.
        """
        # 1. Convert the original image into patches to create the ground truth
        #    This is the inverse of the decoder's final prediction layer.
        target_patches = rearrange(original_image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                   p1=patch_size, p2=patch_size)
        
        # 2. Calculate the element-wise loss for all patches
        loss_all_patches = (reconstruction_patches - target_patches) ** 2
        
        # 3. We only care about the mean loss on the MASKED patches.
        #    The mask has shape (B, num_patches). We need to unsqueeze it to (B, num_patches, 1)
        #    to broadcast correctly with the loss tensor of shape (B, num_patches, patch_features).
        loss_masked = (loss_all_patches * mask.unsqueeze(-1)).sum()
        
        # 4. Normalize the loss by the number of masked patches.
        #    This makes the loss independent of the mask ratio.
        num_masked_patches = mask.sum()
        if num_masked_patches == 0:
            return loss_masked # Should be 0
            
        mean_loss = loss_masked / num_masked_patches
        
        return mean_loss

class L1ReconstructionLoss(nn.Module):
    """
    Alternative reconstruction loss using L1 (MAE) instead of MSE, for MAE outputs.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, reconstruction_patches, original_image, patch_size, mask):
        """
        Calculate L1 loss on the masked patches.
        """
        target_patches = rearrange(original_image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                   p1=patch_size, p2=patch_size)
        
        loss_all_patches = torch.abs(reconstruction_patches - target_patches)
        loss_masked = (loss_all_patches * mask.unsqueeze(-1)).sum()
        
        num_masked_patches = mask.sum()
        if num_masked_patches == 0:
            return loss_masked
            
        mean_loss = loss_masked / num_masked_patches
        return mean_loss