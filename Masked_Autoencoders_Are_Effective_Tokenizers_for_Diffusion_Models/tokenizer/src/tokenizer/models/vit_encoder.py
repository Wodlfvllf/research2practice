import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import *
from .rope_embeddings import *





class Encoder(nn.Module):
    def __init__(self,
                img_size=64, 
                patch_size=4,
                embed_dim=128,
                encoder_depth=4,
                hidden_token_length=16,
                encoder_heads=4,
                in_channels=3,
                mlp_ratio=4.0,
                ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.hidden_token_length = hidden_token_length
        self.encoder_heads = encoder_heads
        self.in_channels = in_channels
        self.mlp_ratio = mlp_ratio
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.n_patches
        self.latent_tokens = nn.Parameter(torch.zeros(1, self.hidden_token_length, self.embed_dim), requires_grad=True)
        
        # Calculate max height/width for RoPE
        max_height = max_width = img_size // patch_size
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, encoder_heads, 
                           max_seq_len=self.num_patches + hidden_token_length,
                           mlp_ratio=mlp_ratio, latent_vec_len=hidden_token_length,
                           max_height=max_height, max_width=max_width) 
            for _ in range(encoder_depth)
        ])
        self.norm_layer = nn.LayerNorm(embed_dim)
        
        # Initialize latent tokens
        nn.init.normal_(self.latent_tokens, std=0.02)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        bs = x.shape[0]
        
        expanded_latent_tokens = self.latent_tokens.expand(bs, -1, -1)
        combined_input = torch.cat([expanded_latent_tokens, x], dim=1)
        
        height = width = self.img_size // self.patch_size
        
        latents = combined_input[:, :self.hidden_token_length, :]
        patches = combined_input[:, self.hidden_token_length:, :]
        
        height = width = self.img_size // self.patch_size
        
        latents = combined_input[:, :self.hidden_token_length, :]
        patches = combined_input[:, self.hidden_token_length:, :]
        
        for block in self.blocks:
            patches = block(patches, height=height, width=width)
            
        combined_output = torch.cat([latents, patches], dim=1)
        combined_output = self.norm_layer(combined_output)
        return combined_output
            
        combined_output = torch.cat([latents, patches], dim=1)
        combined_output = self.norm_layer(combined_output)
        return combined_output
        
        


