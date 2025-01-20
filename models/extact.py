import torch
import torch.nn as nn
import torch.nn.functional as F

# Very similar to GeGLU or SwiGLU, there's a learned gate FN, uses arctan as the activation fn.
class xATGLU(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        # GATE path | VALUE path
        self.proj = nn.Linear(input_dim, output_dim * 2, bias=bias)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='linear')
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.half_pi = torch.pi / 2
        self.inv_pi = 1 / torch.pi
        
    def forward(self, x):
        projected = self.proj(x)
        gate_path, value_path = projected.chunk(2, dim=-1)
        
        # Apply arctan gating with expanded range via learned alpha -- https://arxiv.org/pdf/2405.20768
        gate = (torch.arctan(gate_path) + self.half_pi) * self.inv_pi
        expanded_gate = gate * (1 + 2 * self.alpha) - self.alpha
        
        return expanded_gate * value_path  # g(x) × y