# building a multi head latent attention
# Embedding dimensions: d_model=8
# KV cache dimension: kv _latent_dim = 4
# Number of heads: n_heads = 2
# Head dimension: dh = d_model / n_heads = 8/2 = 4
# Input token embedding: x = [0.1 , -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]4
import torch
import torch.nn as nn
import torch.nn.functional as F

class RopelessMLA(nn.Module):
    def __init__(self, d_model, n_heads, kv_latent_dim):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model / n_heads # dimension per head

        # Projection layers
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.W_dkv = nn.Linear(d_model, kv_latent_dim, bias=False) # Compress into latent KV space
        self.W_uk = nn.Linear(kv_latent_dim, d_model, bias=False) # Decompress k
        self.W_uv = nn.Linear(kv_latent_dim, d_model, bias=False) # Decompress V
        self.W_o = nn.Linear(d_model, d_model, bias=False) # Final output projection

        self.ln = nn.LayerNorm(kv_latent_dim)
        self.register_buffer('absorbed_k', None) # Holds W_q @ W_uk
    
    def forward(self, x, kv_cache=None, past_length=0):
        B, S, D = x.size()

        # Compute absorbed_k once:
        if self.absorbed_k is None:
            # torch.matmul multiply matrices, the first with the second.T (transpose)
            absorbed = torch.matmul(self.W_q.weight, self.W_uk.weight) # (D, latent_dim)
            self.absorbed_k = absorbed.view(self.n_heads, self.dh, -1) # (n_heads, dj, latent_dim)

        # Compress x inot latent KV space 
        new_c_kv = self.ln(self.W_dkv(x)) # (B, S, latent_dim)
        if kv_cache is None:
            c_kv = new_c_kv
        else:
            c_kv = torch.cat([kv_cache, new_c_kv], dim=1) # (B, S_total, latent_dim)
