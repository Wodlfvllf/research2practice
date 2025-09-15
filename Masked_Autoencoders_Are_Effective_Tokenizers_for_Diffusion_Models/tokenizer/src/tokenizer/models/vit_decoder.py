import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import *
from .rope_embeddings import *


class Decoder(nn.Module):
    def __init__(self,
                img_size=64, 
                patch_size=4,
                embed_dim=128,
                decoder_depth=4,
                hidden_token_length=16,
                decoder_heads=4,
                in_channels=3,
                mlp_ratio=4.0,
                ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_depth = decoder_depth
        self.hidden_token_length = hidden_token_length
        self.decoder_heads = decoder_heads
        self.in_channels = in_channels
        self.mlp_ratio = mlp_ratio
        self.num_patches = (img_size // patch_size) ** 2
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        
        # Calculate max height/width for RoPE
        max_height = max_width = img_size // patch_size
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, decoder_heads,
                           max_seq_len=self.num_patches + hidden_token_length,
                           mlp_ratio=mlp_ratio, latent_vec_len=hidden_token_length,
                           max_height=max_height, max_width=max_width)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, embed_dim)
        
        # Initialize mask token
        nn.init.normal_(self.mask_token, std=0.02)
        
    def forward(self, x, ids_restore):
        # x: encoded latent + visible patches from encoder
        # Apply decoder embedding
        x = self.decoder_embed(x)
        
        # Extract latent tokens and visible patches
        latent_tokens = x[:, :self.hidden_token_length, :]
        visible_patches = x[:, self.hidden_token_length:, :]
        
        bs, _, embed_dim = x.shape
        
        # Add mask tokens for missing patches
        mask_tokens = self.mask_token.expand(bs, self.num_patches - visible_patches.shape[1], -1)
        all_patches = torch.cat([visible_patches, mask_tokens], dim=1)
        
        # Unshuffle patches to original order
        all_patches = torch.gather(all_patches, dim=1, 
                                 index=ids_restore.unsqueeze(-1).expand(-1, -1, embed_dim))
        
        # Combine latent tokens with all patches
        decoder_input = torch.cat([latent_tokens, all_patches], dim=1)
        
        height = width = self.img_size // self.patch_size
        
        # Process latent and patch tokens separately through decoder blocks
        latents = decoder_input[:, :self.hidden_token_length, :]
        patches = decoder_input[:, self.hidden_token_length:, :]
        
        for block in self.decoder_blocks:
            patches = block(patches, height=height, width=width)
            
        # Recombine and normalize
        decoder_output = torch.cat([latents, patches], dim=1)
        decoder_output = self.decoder_norm(decoder_output)
        
        # Only predict on patch tokens (skip latent tokens)
        patch_tokens = decoder_output[:, self.hidden_token_length:, :]
        reconstruction = self.decoder_pred(patch_tokens)
        
        return reconstruction