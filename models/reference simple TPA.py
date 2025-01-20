import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .extact import xATGLU

# Modified, original from https://github.com/tensorgi/T6/blob/main/model/T6_ropek.py

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CPLinear(nn.Module):
    def __init__(self, in_features, n_head, head_dim, rank=1, q_rank=12):
        super(CPLinear, self).__init__()
        self.in_features = in_features
        self.n_head = n_head
        self.head_dim = head_dim
        self.rank = rank
        self.q_rank = q_rank

        self.c_q = nn.Linear(in_features, n_head * head_dim, bias=False)
        self.W_A_k = nn.Linear(in_features, n_head * rank, bias=False)
        self.W_A_v = nn.Linear(in_features, n_head * rank, bias=False)
        self.W_B_k = nn.Linear(in_features, rank * head_dim, bias=False)
        self.W_B_v = nn.Linear(in_features, rank * head_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        W_A_k_tensor = self.W_A_k.weight.view(self.in_features, self.n_head, self.rank)
        W_A_v_tensor = self.W_A_v.weight.view(self.in_features, self.n_head, self.rank)
        nn.init.xavier_uniform_(W_A_k_tensor)
        nn.init.xavier_uniform_(W_A_v_tensor)
        self.W_A_k.weight.data = W_A_k_tensor.view_as(self.W_A_k.weight)
        self.W_A_v.weight.data = W_A_v_tensor.view_as(self.W_A_v.weight)

        W_B_k_tensor = self.W_B_k.weight.view(self.in_features, self.rank, self.head_dim)
        W_B_v_tensor = self.W_B_v.weight.view(self.in_features, self.rank, self.head_dim)
        nn.init.xavier_uniform_(W_B_k_tensor)
        nn.init.xavier_uniform_(W_B_v_tensor)
        self.W_B_k.weight.data = W_B_k_tensor.view_as(self.W_B_k.weight)
        self.W_B_v.weight.data = W_B_v_tensor.view_as(self.W_B_v.weight)
        
        nn.init.xavier_uniform_(self.c_q.weight)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        
        A_k = self.W_A_k(x).view(batch_size, seq_len, self.n_head, self.rank)
        A_v = self.W_A_v(x).view(batch_size, seq_len, self.n_head, self.rank)
        
        B_k = self.W_B_k(x).view(batch_size, seq_len, self.rank, self.head_dim)
        B_v = self.W_B_v(x).view(batch_size, seq_len, self.rank, self.head_dim)

        A_k = A_k.view(batch_size * seq_len, self.n_head, self.rank)
        A_v = A_v.view(batch_size * seq_len, self.n_head, self.rank)
        
        B_k = B_k.view(batch_size * seq_len, self.rank, self.head_dim)
        B_v = B_v.view(batch_size * seq_len, self.rank, self.head_dim)
        
        k = torch.bmm(A_k, B_k).div_(self.rank).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = torch.bmm(A_v, B_v).div_(self.rank).view(batch_size, seq_len, self.n_head, self.head_dim)

        return q, k, v

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, rank=1, q_rank=12, using_groupnorm=False):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.n_embd = n_embd
        
        self.c_qkv = CPLinear(n_embd, n_head, head_dim, rank, q_rank)
        self.c_proj = nn.Linear(n_head * head_dim, n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(head_dim)
        
        self.using_groupnorm = using_groupnorm
        if self.using_groupnorm:
            self.subln = RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_qkv(x)
        
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True
        )
        
        if self.using_groupnorm:
            y = self.subln(y)
        
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = math.floor(8 / 3 * n_embd)
        
        self.c_fc1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)
        x = F.silu(x1) * x2
        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, rank=2, q_rank=6, using_groupnorm=True):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, head_dim, rank, q_rank, using_groupnorm)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x