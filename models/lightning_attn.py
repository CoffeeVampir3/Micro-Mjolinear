import torch
import torch.nn as nn
from lightning_attn.ops import lightning_attn_func
from lightning_attn.utils import build_slope_tensor

# For example only.
class MinimalLightningAttention2(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.slopes = build_slope_tensor(num_heads)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        slopes = self.slopes.to(q.device).to(torch.float32)
        attn = lightning_attn_func(q, k, v, slopes)
        
        out = attn.transpose(1, 2).reshape(B, N, C)
        return self.out(out)