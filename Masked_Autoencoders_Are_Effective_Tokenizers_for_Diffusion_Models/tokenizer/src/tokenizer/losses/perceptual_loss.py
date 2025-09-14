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


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features (classic approach).
    Compares features from multiple layers of a pre-trained VGG network.
    """
    
    def __init__(self, 
                 model_name='vgg16_bn',
                 layers=['features.7', 'features.14', 'features.24', 'features.34'],
                 layer_weights=[1.0, 1.0, 1.0, 1.0],
                 normalize_input=True,
                 reduction='mean'):
        """
        Args:
            model_name (str): VGG model name from timm
            layers (list): Layer names to extract features from
            layer_weights (list): Weights for each layer's contribution
            normalize_input (bool): Whether to normalize inputs to ImageNet stats
            reduction (str): Reduction method for loss
        """
        super().__init__()
        
        # Load pre-trained VGG model
        self.model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.model.eval()
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.layers = layers
        self.layer_weights = layer_weights
        self.reduction = reduction
        
        # ImageNet normalization
        if normalize_input:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize = nn.Identity()
    
    def extract_features(self, x):
        """Extract features from specified layers."""
        # Normalize input
        x = self.normalize(x)
        
        # Extract features
        features = []
        for feature_map in self.model(x):
            features.append(feature_map)
        
        return features
    
    def forward(self, reconstructed, original, mask):
        """
        Calculate perceptual loss between reconstructed and original images.
        
        Args:
            reconstructed (torch.Tensor): Reconstructed images (B, C, H, W)
            original (torch.Tensor): Original images (B, C, H, W)
        
        Returns:
            torch.Tensor: Perceptual loss
        """
        # Extract features from both images
        features_reconstructed = self.extract_features(reconstructed)
        features_original = self.extract_features(original)
        
        # Calculate loss for each layer
        total_loss = 0.0
        for i, (feat_recon, feat_orig, weight) in enumerate(
            zip(features_reconstructed, features_original, self.layer_weights)):
            
            layer_loss = F.l1_loss(feat_recon, feat_orig, reduction=self.reduction, mask=mask)
            total_loss += weight * layer_loss
        
        return total_loss

