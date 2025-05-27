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
        self.dh = d_model // n_heads # dimension per head

        # Projection layers
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.W_dkv = nn.Linear(d_model, kv_latent_dim, bias=False) # Compress into latent KV space
        self.W_uk = nn.Linear(kv_latent_dim, d_model, bias=False) # Decompress k
        self.W_uv = nn.Linear(kv_latent_dim, d_model, bias=False) # Decompress V
        self.W_o = nn.Linear(d_model, d_model, bias=False) # Final output projection

        self.ln = nn.LayerNorm(kv_latent_dim)
        self.register_buffer('absorbed_k', None) # Holds W_q @ W_uk
    
    def forward(self, x, kv_cache=None, past_length=0):
        # the x is the new token coming in
        B, S, D = x.size()

        # Compute absorbed_k once:
        if self.absorbed_k is None:
            # torch.matmul multiply matrices, the first with the second.T (transpose)
            absorbed = torch.matmul(self.W_q.weight, self.W_uk.weight) # (D, latent_dim)
            # in the following line we tell to use the absorbed matrix and to group by the number of heads. head dimension and latent dimension
            self.absorbed_k = absorbed.view(self.n_heads, self.dh, -1) # (n_heads, dh, latent_dim)

        # Compress x inot latent KV space 
        new_c_kv = self.ln(self.W_dkv(x)) # (B, S, latent_dim)       The .ln just does layer normalization
        if kv_cache is None:
            c_kv = new_c_kv
        else: # this new vector get appended to the cache
            c_kv = torch.cat([kv_cache, new_c_kv], dim=1) # (B, S_total, latent_dim)

        S_full = c_kv.size(1)
        # decompress V to full d_model and split into heads
        v_full = self.W_uv(c_kv) # (B, S_full, D)
        v = v_full.view(B, S_full, self.n_heads, self.dh).transpose(1,2) # (B, n_heads, S_full, )
        # Use input x directly (since W_q is absorbed)
        q = x.view(B, S, self.n_heads, self.dh) # (B, S, n_heads, dh)
        # Compute attention scores
        attn_scores = torch.zeros(B, self.n_heads, S, S_full, device=x.device)
        for h in range(self.n_heads):
            tmp = torch.matmul(q[:, :, h], self.absorbed_k[h])
            attn_scores[:, h] = torch.bmm(tmp, c_kv.transpose(1,2))        

        # Scale and apply causal mask
        attn_scores = attn_scores / (self.dh ** 0.5)
        mask = torch.trill(torch.ones((S, S_full), device = x.device), diagonal=past_length)
        attn_scores = attn_scores.masked_fill(mask.view(1,1,S,S_full) == 0, float('-i'))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, n_heads, S, S_full)

        # Apply attention weights to each head's V separately
        out_heads = []
        for h in range(self.n_heads):
            context_h = torch.matmul(attn_weights[:, h], v[:, h]) # (B, S, dh)
            out_heads.append(context_h)

        # Concatenate all heads outputs along the feature dimension
        out = torch.cat(out_heads, dim=-1) # (B, S, D)

        return self.W_o(out), c_kv # Final output projection + updated latent cache
