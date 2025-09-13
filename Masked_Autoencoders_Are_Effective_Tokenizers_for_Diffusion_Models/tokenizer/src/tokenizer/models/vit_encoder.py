import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# ---------- Helper: precompute cos & sin for RoPE ----------
def precompute_cos_sin(max_seq_len: int, head_dim: int, base: float = 10000.0, device=None, dtype=torch.float32):
    """
    Return cos, sin of shape (max_seq_len, head_dim//2) as float32 on `device`.
    head_dim must be even.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE (paired dims)"
    m = head_dim // 2
    device = device if device is not None else torch.device('cpu')
    inv_freq = 1.0 / (base ** (torch.arange(0, m, device=device, dtype=dtype) * 2.0 / head_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # (max_seq_len, m)
    return torch.cos(angles), torch.sin(angles)  # both (max_seq_len, m)


def precompute_2d_cos_sin(max_height: int, max_width: int, head_dim: int, base: float = 10000.0, device=None, dtype=torch.float32):
    """
    Precompute 2D RoPE for spatial positions (x, y coordinates).
    Returns cos_x, sin_x, cos_y, sin_y for 2D spatial encoding.
    Each has shape (max_height, max_width, head_dim//4).
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    m = head_dim // 4  # Each spatial dim gets head_dim//4 frequencies
    device = device if device is not None else torch.device('cpu')
    
    inv_freq = 1.0 / (base ** (torch.arange(0, m, device=device, dtype=dtype) * 4.0 / head_dim))
    
    # Create position grids
    y_pos = torch.arange(max_height, device=device, dtype=dtype)
    x_pos = torch.arange(max_width, device=device, dtype=dtype)
    
    # Compute angles for x and y
    x_angles = x_pos.unsqueeze(1) * inv_freq.unsqueeze(0)  # (max_width, m)
    y_angles = y_pos.unsqueeze(1) * inv_freq.unsqueeze(0)  # (max_height, m)
    
    # Expand to full grid
    cos_x = torch.cos(x_angles).unsqueeze(0).expand(max_height, -1, -1)  # (max_height, max_width, m)
    sin_x = torch.sin(x_angles).unsqueeze(0).expand(max_height, -1, -1)
    cos_y = torch.cos(y_angles).unsqueeze(1).expand(-1, max_width, -1)   # (max_height, max_width, m)
    sin_y = torch.sin(y_angles).unsqueeze(1).expand(-1, max_width, -1)
    
    return cos_x, sin_x, cos_y, sin_y


def apply_rope_real(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to tensor x using precomputed cos/sin.

    x: shape (..., seq_len, head_dim) where head_dim = 2*m.
       Example shapes used in this code: (B, n_heads, S, head_dim)
    cos/sin: (max_seq_len, m) float32 (we'll cast them to x.dtype & x.device as needed)

    Returns: same shape as x with rotary embedding applied across last dim pairs.
    """
    orig_dtype = x.dtype
    seq_len = x.shape[-2]
    head_dim = x.shape[-1]
    assert head_dim % 2 == 0
    m = head_dim // 2

    # ensure cos/sin have enough length along seq axis
    cos = cos[:seq_len].to(device=x.device, dtype=orig_dtype)   # (seq_len, m)
    sin = sin[:seq_len].to(device=x.device, dtype=orig_dtype)

    # reshape x to (..., seq_len, m, 2)
    x_ = x.reshape(*x.shape[:-1], m, 2)   # last dimension pairs
    a = x_[..., 0]   # (..., seq_len, m)
    b = x_[..., 1]   # (..., seq_len, m)

    # Broadcast cos/sin: cos shape (seq_len, m) -> (1, seq_len, m) and then broadcast to match a/b
    # Because a/b have arbitrary leading dims, broadcasting rules align last two dims.
    cos_ = cos.unsqueeze(0)  # (1, seq_len, m)
    sin_ = sin.unsqueeze(0)

    new_a = a * cos_ - b * sin_
    new_b = a * sin_ + b * cos_
    x_rot = torch.stack((new_a, new_b), dim=-1)  # (..., seq_len, m, 2)
    out = x_rot.reshape(*x.shape[:-2], seq_len, head_dim)
    return out


def apply_2d_rope(x: torch.Tensor, cos_x: torch.Tensor, sin_x: torch.Tensor, 
                  cos_y: torch.Tensor, sin_y: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Apply 2D RoPE to image patch tokens.
    
    x: (..., seq_len, head_dim) where seq_len = height * width
    cos_x, sin_x, cos_y, sin_y: (max_height, max_width, head_dim//4)
    """
    orig_dtype = x.dtype
    seq_len = x.shape[-2]
    head_dim = x.shape[-1]
    
    assert seq_len == height * width, f"seq_len {seq_len} must equal height*width {height*width}"
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    
    # Get spatial encodings for current image size
    cos_x = cos_x[:height, :width].to(device=x.device, dtype=orig_dtype)  # (h, w, m)
    sin_x = sin_x[:height, :width].to(device=x.device, dtype=orig_dtype)
    cos_y = cos_y[:height, :width].to(device=x.device, dtype=orig_dtype)
    sin_y = sin_y[:height, :width].to(device=x.device, dtype=orig_dtype)
    
    # Flatten spatial dimensions
    cos_x = cos_x.reshape(seq_len, -1)  # (seq_len, m)
    sin_x = sin_x.reshape(seq_len, -1)
    cos_y = cos_y.reshape(seq_len, -1)
    sin_y = sin_y.reshape(seq_len, -1)
    
    # Split head_dim into 4 parts: x_real, x_imag, y_real, y_imag
    quarter_dim = head_dim // 4
    x_real = x[..., :quarter_dim]                    # (..., seq_len, quarter_dim)
    x_imag = x[..., quarter_dim:2*quarter_dim]
    y_real = x[..., 2*quarter_dim:3*quarter_dim]
    y_imag = x[..., 3*quarter_dim:]
    
    # Apply rotation for x coordinates
    new_x_real = x_real * cos_x.unsqueeze(0) - x_imag * sin_x.unsqueeze(0)
    new_x_imag = x_real * sin_x.unsqueeze(0) + x_imag * cos_x.unsqueeze(0)
    
    # Apply rotation for y coordinates  
    new_y_real = y_real * cos_y.unsqueeze(0) - y_imag * sin_y.unsqueeze(0)
    new_y_imag = y_real * sin_y.unsqueeze(0) + y_imag * cos_y.unsqueeze(0)
    
    # Concatenate back
    out = torch.cat([new_x_real, new_x_imag, new_y_real, new_y_imag], dim=-1)
    return out


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

        return out


# ---------- Small MLP, PatchEmbedding, TransformerBlock (unchanged except attention) ----------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, 64)
        self.up_proj = nn.Linear(64, input_dim)
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
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (batch, channels, H, W)
        x = self.proj(x)  # (batch, embed_dim, H/patch, W/patch)
        x = x.flatten(2)  # (batch, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, max_seq_len=512):
        super().__init__()
        self.attention = Attention(hidden_dim, n_heads, max_seq_len=max_seq_len)
        self.mlp = MLP(hidden_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.attention(x)
        x = self.relu(x)
        x = self.mlp(x)
        x = self.relu(x)
        x = self.norm(x)
        return x

class Encoder(nn.Module):
    def __init__(self,
                img_size = 64, 
                patch_size = 4,
                embed_dim = 128,
                encoder_depth = 4,
                hidden_token_length = 16,
                encoder_heads = 4,
                in_channels = 3,
                ):
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.hidden_token_length = hidden_token_length
        self.mlp_ratio = mlp_ratio
        self.encoder_heads = encoder_heads
        self.decoder_heads = decoder_heads
        self.in_channels = in_channels
        
        self.patch_embed = PatchEmbedding(self.img_size, self.patch_size, self.in_channels, self.embed_dim)
        self.num_patches = self.patch_embed.n_patches
        self.latent_tokens = nn.Parameter(torch.zeros(1, self.hidden_token_length, self.embed_dim), requires_grad = True)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.hidden_dim, self.n_heads) for _ in range(self.encoder_depth)
        ])
        self.norm_layer = nn.LayerNorm()
        
    def forward(self, x):
        bs, seq_len, hid_dim = x.shape
        expanded_latent_tokens = self.latent_tokens.expand(bs, -1, -1)
        combined_input = torch.cat([expanded_latent_tokens, x], dim = 1)
        for block in self.blocks:
            combined_input = block(combined_input)
            
        return combined_input
        
class Decoder(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass
    
class Model(nn.Module):
    def __init__(self, 
                img_size = 64, 
                patch_size = 4,
                embed_dim = 128,
                encoder_depth = 4,
                decoder_depth = 4,
                hidden_token_length = 16,
                mlp_ratio = 0.4,
                encoder_heads = 4,
                decoder_heads = 4,
                in_channels = 3,
                ):
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.hidden_token_length = hidden_token_length
        self.mlp_ratio = mlp_ratio
        self.encoder_heads = encoder_heads
        self.decoder_heads = decoder_heads
        self.in_channels = in_channels
        
        #-------------------------------------------------------------------------------------
        # Encoder Specifics
        self.encoder = Encoder(
            img_size = img_size
            patch_size = patch_size
            embed_dim = embed_dim
            encoder_depth = encoder_depth
            hidden_token_length = hidden_token_length
            encoder_heads = encoder_heads
            in_channels = in_channels
        )
        
        #--------------------------------------------------------------------------------------
        # Decoder Specifics
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True)
        
    