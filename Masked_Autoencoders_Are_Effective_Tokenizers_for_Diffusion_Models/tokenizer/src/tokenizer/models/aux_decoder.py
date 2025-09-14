
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import torchvision.transforms as transforms
from skimage.feature import hog
import numpy as np
from .attention import TransformerBlock

class AuxiliaryDecoder(nn.Module):
    """Auxiliary decoder that matches your main decoder architecture"""
    def __init__(self, 
                 embed_dim=128,
                 output_dim=81,  # Feature dimension (81 for HOG, 1024 for CLIP)
                 decoder_depth=4,
                 decoder_heads=4,
                 hidden_token_length=16,
                 mlp_ratio=4.0,
                 img_size=64,
                 patch_size=4):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.decoder_depth = decoder_depth
        self.hidden_token_length = hidden_token_length
        self.decoder_heads = decoder_heads
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Same structure as your main decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        
        # Calculate max height/width for RoPE (same as your decoder)
        max_height = max_width = img_size // patch_size
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, decoder_heads,
                           max_seq_len=self.num_patches + hidden_token_length,
                           mlp_ratio=mlp_ratio, latent_vec_len=hidden_token_length,
                           max_height=max_height, max_width=max_width)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(embed_dim)
        # Different prediction head for auxiliary features
        self.decoder_pred = nn.Linear(embed_dim, output_dim)
        
        # Initialize mask token (same as your decoder)
        nn.init.normal_(self.mask_token, std=0.02)
        
    def forward(self, x, ids_restore):
        """
        Exactly the same structure as your main decoder, but predicts features instead of pixels
        x: encoded latent + visible patches from encoder
        """
        # Apply decoder embedding (same as your decoder)
        x = self.decoder_embed(x)
        
        # Extract latent tokens and visible patches (same as your decoder)
        latent_tokens = x[:, :self.hidden_token_length, :]
        visible_patches = x[:, self.hidden_token_length:, :]
        
        bs, _, embed_dim = x.shape
        
        # Add mask tokens for missing patches (same as your decoder)
        mask_tokens = self.mask_token.expand(bs, self.num_patches - visible_patches.shape[1], -1)
        all_patches = torch.cat([visible_patches, mask_tokens], dim=1)
        
        # Unshuffle patches to original order (same as your decoder)
        all_patches = torch.gather(all_patches, dim=1, 
                                 index=ids_restore.unsqueeze(-1).expand(-1, -1, embed_dim))
        
        # Combine latent tokens with all patches (same as your decoder)
        decoder_input = torch.cat([latent_tokens, all_patches], dim=1)
        
        height = width = self.img_size // self.patch_size
        
        # Forward through transformer blocks (same as your decoder)
        for block in self.decoder_blocks:
            decoder_input = block(decoder_input, height=height, width=width)
            
        decoder_input = self.decoder_norm(decoder_input)
        
        # Only predict on patch tokens (skip latent tokens) - same as your decoder
        patch_tokens = decoder_input[:, self.hidden_token_length:, :]
        feature_prediction = self.decoder_pred(patch_tokens)  # Predict features instead of pixels
        
        return feature_prediction

class FeatureExtractors:
    """Extract different types of features for auxiliary targets"""
    
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize CLIP for semantic features
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
    def extract_hog_features(self, images, patch_size=4):
        """Extract HOG features for each patch"""
        batch_size, channels, height, width = images.shape
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        
        hog_features = []
        for img in images:
            # Convert to numpy and handle normalization
            img_np = img.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1.0:  # If normalized to [0,1]
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            else:
                img_np = img_np.clip(0, 255).astype(np.uint8)
            
            # Convert to grayscale
            if len(img_np.shape) == 3:
                img_gray = np.mean(img_np, axis=2)
            else:
                img_gray = img_np
            
            patch_hog = []
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    patch = img_gray[i*patch_size:(i+1)*patch_size, 
                                   j*patch_size:(j+1)*patch_size]
                    try:
                        # For small patches, use smaller parameters
                        hog_feat = hog(patch, 
                                     orientations=9, 
                                     pixels_per_cell=(2, 2),  # Small for 4x4 patches
                                     cells_per_block=(1, 1),  # Single cell per block
                                     visualize=False)
                        patch_hog.append(hog_feat)
                    except Exception as e:
                        # Fallback: create zero features with expected size
                        # For 4x4 patch with (2,2) pixels_per_cell, we get 2x2=4 cells
                        # With 9 orientations, that's 4*9=36 features per cell
                        # With (1,1) cells_per_block, we get 36 features
                        patch_hog.append(np.zeros(36))
            hog_features.append(patch_hog)
        
        return torch.tensor(hog_features, dtype=torch.float32, device=self.device)
    
    def extract_clip_features(self, images):
        """Extract CLIP visual features"""
        with torch.no_grad():
            # Resize images to CLIP expected size (224x224)
            transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True)
            ])
            
            images_resized = torch.stack([transform(img) for img in images])
            
            # Ensure images are in correct format for CLIP
            if images_resized.max() <= 1.0:
                images_for_clip = images_resized
            else:
                images_for_clip = images_resized / 255.0
            
            # Convert to PIL format expected by CLIP processor
            images_pil = []
            for img in images_for_clip:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                images_pil.append(img_np)
            
            inputs = self.clip_processor(images=images_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get patch features from CLIP vision encoder
            vision_outputs = self.clip_model.vision_model(**inputs)
            patch_features = vision_outputs.last_hidden_state  # [batch, num_patches+1, dim]
            
            # Remove CLS token
            clip_patches = patch_features[:, 1:, :]  # Remove CLS token
            
            # CLIP has 196 patches (14x14), we need to match our patch count
            target_patches = (images.shape[2] // 4) ** 2  # Your patch_size=4
            if clip_patches.shape[1] != target_patches:
                # Interpolate to match patch count
                if target_patches < clip_patches.shape[1]:
                    # Downsample
                    clip_patches = clip_patches[:, :target_patches, :]
                else:
                    # Upsample using interpolation
                    # Reshape for interpolation
                    B, N, D = clip_patches.shape
                    clip_h = clip_w = int(N ** 0.5)  # 14 for CLIP
                    target_h = target_w = int(target_patches ** 0.5)
                    
                    clip_patches = clip_patches.reshape(B, clip_h, clip_w, D)
                    clip_patches = clip_patches.permute(0, 3, 1, 2)  # B, D, H, W
                    clip_patches = F.interpolate(clip_patches, size=(target_h, target_w), mode='bilinear')
                    clip_patches = clip_patches.permute(0, 2, 3, 1).reshape(B, target_patches, D)
            
        return clip_patches
