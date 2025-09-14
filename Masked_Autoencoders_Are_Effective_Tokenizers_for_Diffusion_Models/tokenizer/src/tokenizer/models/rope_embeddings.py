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
