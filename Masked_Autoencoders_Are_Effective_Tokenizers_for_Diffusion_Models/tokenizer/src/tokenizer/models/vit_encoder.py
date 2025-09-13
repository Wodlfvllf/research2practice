import torch
import torch.nn as nn
from einops import rearrage

class Attention(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Define Q, K, V matrices here
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        B, Seq_len, _ = x.shape
        
        # Project inputs
        query = self.Q(x)
        key = self.K(x)
        value = self.V(x)
        
        # Split into heads
        Q_new = rearrange(query, 'b s (n h) -> b n s h', n=self.n_heads)
        K_new = rearrange(key, 'b s (n h) -> b n s h', n=self.n_heads)
        V_new = rearrange(value, 'b s (n h) -> b n s h', n=self.n_heads)
        
        # Compute attention scores
        attn = Q_new @ rearrange(K_new, 'b n s h -> b n h s')  # shape: (b, n, s, s)
        attn = attn / np.sqrt(self.head_dim)                   # scale by sqrt(d_k)
        
        # Softmax
        score = F.softmax(attn, dim=-1)                        # (b, n, s, s)
        
        # Weighted sum
        output = score @ V_new                                 # (b, n, s, h)
        
        # Merge heads
        out = rearrange(output, 'b n s h -> b s (n h)')
        
        return out
    
    
class 