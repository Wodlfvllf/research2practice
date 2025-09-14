import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .rope_embeddings import precompute_cos_sin, precompute_2d_cos_sin, apply_rope_real, apply_2d_rope

# ---------- Attention with RoPE ----------
class Attention(nn.Module):
    def __init__(self, hidden_dim, n_heads, max_seq_len: int = 512, rope_base: float = 10000.0, 
                 latent_vec_len: int = 16, max_height: int = 32, max_width: int = 32):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE (pairs)"
        
        # For 2D RoPE on image patches, we need head_dim divisible by 4
        self.spatial_head_dim = self.head_dim - latent_vec_len
        if self.spatial_head_dim > 0:
            assert self.spatial_head_dim % 4 == 0, "spatial portion of head_dim must be divisible by 4 for 2D RoPE"

        # Projection matrices
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.latent_vec_len = latent_vec_len
        
        # Precompute 1D RoPE for latent tokens (temporal/sequential)
        cos_1d, sin_1d = precompute_cos_sin(max_seq_len, latent_vec_len, base=rope_base, 
                                           device=torch.device('cpu'), dtype=torch.float32)
        self.register_buffer('rope_cos_1d', cos_1d, persistent=False)
        self.register_buffer('rope_sin_1d', sin_1d, persistent=False)
        
        # Precompute 2D RoPE for spatial tokens (image patches)
        if self.spatial_head_dim > 0:
            cos_x, sin_x, cos_y, sin_y = precompute_2d_cos_sin(max_height, max_width, self.spatial_head_dim, 
                                                              base=rope_base, device=torch.device('cpu'), 
                                                              dtype=torch.float32)
            self.register_buffer('rope_cos_x', cos_x, persistent=False)
            self.register_buffer('rope_sin_x', sin_x, persistent=False)
            self.register_buffer('rope_cos_y', cos_y, persistent=False)
            self.register_buffer('rope_sin_y', sin_y, persistent=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, height: int = None, width: int = None):
        """
        x: (B, Seq_len, hidden_dim)
        height, width: spatial dimensions for 2D RoPE on image patches
        """
        B, S, _ = x.shape

        # Project
        q = self.Q(x)  # (B, S, hidden_dim)
        k = self.K(x)
        v = self.V(x)

        # Split into heads -> (B, n_heads, S, head_dim)
        Qh = rearrange(q, 'b s (n h) -> b n s h', n=self.n_heads)
        Kh = rearrange(k, 'b s (n h) -> b n s h', n=self.n_heads)
        Vh = rearrange(v, 'b s (n h) -> b n s h', n=self.n_heads)

        # Split each head into latent and spatial portions
        latent_query = Qh[..., :self.latent_vec_len]      # (B, n_heads, S, latent_vec_len)
        spatial_query = Qh[..., self.latent_vec_len:]     # (B, n_heads, S, spatial_head_dim)
        
        latent_key = Kh[..., :self.latent_vec_len]
        spatial_key = Kh[..., self.latent_vec_len:]

        # Apply 1D RoPE to latent portions (sequential/temporal encoding)
        if self.latent_vec_len > 0:
            latent_query = apply_rope_real(latent_query, self.rope_cos_1d, self.rope_sin_1d)
            latent_key = apply_rope_real(latent_key, self.rope_cos_1d, self.rope_sin_1d)

        # Apply 2D RoPE to spatial portions (image patch encoding)
        if self.spatial_head_dim > 0 and height is not None and width is not None:
            spatial_query = apply_2d_rope(spatial_query, self.rope_cos_x, self.rope_sin_x,
                                        self.rope_cos_y, self.rope_sin_y, height, width)
            spatial_key = apply_2d_rope(spatial_key, self.rope_cos_x, self.rope_sin_x,
                                      self.rope_cos_y, self.rope_sin_y, height, width)

        # Recombine latent and spatial portions
        if self.latent_vec_len > 0 and self.spatial_head_dim > 0:
            Qh_rope = torch.cat([latent_query, spatial_query], dim=-1)
            Kh_rope = torch.cat([latent_key, spatial_key], dim=-1)
        elif self.latent_vec_len > 0:
            Qh_rope = latent_query
            Kh_rope = latent_key
        else:
            Qh_rope = spatial_query
            Kh_rope = spatial_key

        # Compute attention scores (B, n_heads, S, S)
        attn = torch.matmul(Qh_rope, Kh_rope.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum -> (B, n_heads, S, head_dim)
        out = torch.matmul(attn, Vh)

        # Merge heads -> (B, S, hidden_dim)
        out = rearrange(out, 'b n s h -> b s (n h)')
        out = self.out_proj(out)

        return out


# ---------- Small MLP, PatchEmbedding, TransformerBlock ----------
class MLP(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4.0):
        super().__init__()
        hidden_dim = int(input_dim * mlp_ratio)
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.down_proj(x)
        out = self.relu(out)
        out = self.up_proj(out)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.height = img_size // patch_size
        self.width = img_size // patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (batch, channels, H, W)
        x = self.proj(x)  # (batch, embed_dim, H/patch, W/patch)
        x = x.flatten(2)  # (batch, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, max_seq_len=512, mlp_ratio=4.0, 
                 latent_vec_len=16, max_height=32, max_width=32):
        super().__init__()
        self.attention = Attention(hidden_dim, n_heads, max_seq_len=max_seq_len,
                                 latent_vec_len=latent_vec_len, max_height=max_height, max_width=max_width)
        self.mlp = MLP(hidden_dim, mlp_ratio=mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, height=None, width=None):
        # Pre-norm residual connection
        x = x + self.attention(self.norm1(x), height=height, width=width)
        x = x + self.mlp(self.norm2(x))
        return x