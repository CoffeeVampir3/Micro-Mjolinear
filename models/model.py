import torch
import torch.nn as nn
import torch.nn.functional as F

from .extact import xATGLU
from .lightning_tensor_product import TensorProductTransformerBlock

class WaifuLMUwU(nn.Module):
    def __init__(self, vocab_size, n_embd=768, n_layer=12, n_head=8, max_seq_length=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        
        self.blocks = nn.ModuleList([
            TensorProductTransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                head_dim=n_embd // n_head,
                rank=2,
                q_rank=6,
                using_groupnorm=True,
                use_lightning=True
            ) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        x = self.token_embedding(idx)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        # For CCE:
        # x -> The normalized hidden states
        # head weight -> The classifier weight matrix
        return x, self.lm_head.weight