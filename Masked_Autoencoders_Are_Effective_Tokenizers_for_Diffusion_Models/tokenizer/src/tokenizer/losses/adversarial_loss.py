import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PatchDiscriminator(nn.Module):
    """
    A patch-based discriminator for adversarial training.
    It tries to classify if a patch is real or fake.
    This is a simplified version of the PatchGAN discriminator.
    """
    def __init__(self, in_channels=3, patch_size=4, base_channels=64, n_layers=3):
        super().__init__()
        
        # The input to the discriminator is a single patch
        # Input shape: (B, C, patch_size, patch_size)
        
        layers = []
        in_dim = in_channels
        out_dim = base_channels
        
        # Initial convolution layer
        layers.append(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Subsequent layers
        for _ in range(n_layers - 1):
            in_dim = out_dim
            out_dim *= 2
            layers.append(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
            )
            layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        # Final layer to produce a single logit
        # The receptive field should cover the patch
        layers.append(
            nn.Conv2d(out_dim, 1, kernel_size=patch_size, stride=1, padding=0)
        )
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: A batch of image patches. Shape: (B, C, patch_size, patch_size)
        Returns logits. Shape: (B, 1)
        """
        logits = self.model(x)
        return logits.view(x.shape[0], -1)


class AdversarialLoss(nn.Module):
    """
    Computes the adversarial loss for both the generator (MAE) and the discriminator.
    """
    def __init__(self, 
                 discriminator,
                 patch_size=4,
                 in_channels=3,
                 reduction='mean'):
        super().__init__()
        self.discriminator = discriminator
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.reduction = reduction
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, reconstruction_patches, original_image, mask):
        """
        Calculates the adversarial loss.
        
        Args:
            reconstruction_patches (torch.Tensor): Output from the MAE decoder.
                                                  Shape: (B, num_patches, patch_features)
            original_image (torch.Tensor): The original input image.
                                           Shape: (B, C, H, W)
            mask (torch.Tensor): The binary mask from the MAE. 
                                 Shape: (B, num_patches), where 1 indicates a masked patch.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Generator loss, Discriminator loss
        """
        # 1. Prepare real and fake patches for the discriminator
        with torch.no_grad():
            real_patches = rearrange(original_image, 'b c (h p1) (w p2) -> (b h w) c p1 p2', 
                                     p1=self.patch_size, p2=self.patch_size)
        
        # Reshape reconstructed patches to image patch format
        num_patches = reconstruction_patches.shape[1]
        fake_patches = reconstruction_patches.view(
            -1, num_patches, self.in_channels, self.patch_size, self.patch_size
        ).view(-1, self.in_channels, self.patch_size, self.patch_size)
        
        # We only compute the loss on the masked patches
        mask_flat = mask.flatten() # Shape: (B * num_patches)
        masked_indices = torch.where(mask_flat == 1)[0]
        
        real_masked_patches = real_patches[masked_indices]
        fake_masked_patches = fake_patches[masked_indices]
        
        # 2. Calculate Discriminator Loss
        # Detach fake patches so we don't update the generator with the discriminator loss
        d_fake_logits = self.discriminator(fake_masked_patches.detach())
        d_real_logits = self.discriminator(real_masked_patches)
        
        d_loss_fake = self.loss_fn(d_fake_logits, torch.zeros_like(d_fake_logits))
        d_loss_real = self.loss_fn(d_real_logits, torch.ones_like(d_real_logits))
        
        d_loss = (d_loss_fake + d_loss_real) * 0.5
        
        # 3. Calculate Generator Loss
        # We want the generator to fool the discriminator
        g_logits = self.discriminator(fake_masked_patches)
        g_loss = self.loss_fn(g_logits, torch.ones_like(g_logits))
        
        return g_loss, d_loss
