"""
Perceptual Loss (L_percep) Implementation
Uses pre-trained feature extractors to compare high-level features between
original and reconstructed images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from einops import rearrange

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features, adapted for MAE.
    The loss is computed on the full reconstructed image, as perceptual features
    are global. Masking is not typically applied here, as the goal is to ensure
    the entire reconstructed image (with visible patches + predicted patches) is coherent.
    """
    
    def __init__(self, 
                 model_name='vgg16_bn',
                 normalize_input=True,
                 reduction='mean'):
        super().__init__()
        
        # Create a feature extractor from a pre-trained model
        self.model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.reduction = reduction
        
        if normalize_input:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            
            self.normalize = nn.Identity()
    
    def forward(self, reconstruction_patches, original_image, patch_size):
        """
        Calculate perceptual loss. For this loss, we need to see the full picture.

        Args:
            reconstruction_patches (torch.Tensor): The output from the MAE decoder.
                                                  Shape: (B, num_patches, patch_features)
            original_image (torch.Tensor): The original input image.
                                           Shape: (B, C, H, W)
            patch_size (int): The size of a single patch.
        
        Returns:
            torch.Tensor: Perceptual loss.
        """
        # 1. Reshape the reconstructed patches back into an image
        reconstructed_image = rearrange(reconstruction_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                        h=original_image.shape[2] // patch_size, 
                                        w=original_image.shape[3] // patch_size,
                                        p1=patch_size, p2=patch_size)
        
        # 2. Normalize both images
        original_norm = self.normalize(original_image)
        reconstructed_norm = self.normalize(reconstructed_image)

        # 3. Extract features
        features_original = self.model(original_norm)
        features_reconstructed = self.model(reconstructed_norm)
        
        # 4. Calculate L1 loss between the feature maps
        total_loss = 0.0
        for feat_orig, feat_recon in zip(features_original, features_reconstructed):
            total_loss += F.l1_loss(feat_recon, feat_orig, reduction=self.reduction)
        
        return total_loss