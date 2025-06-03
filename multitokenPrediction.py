
import torch
import torch.nn as nn
import torch.nn.functional as F

# define RMSNorm class
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps:float = 1e-8):
        super().__init__()
        self.eps  = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True)+ self.eps)
        return x /rms # this is a simple some kind of normalization
    

# define the multi-token prediction class
class SimpleMTP(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, num_heads: int = 3, nhead:int =2):
        """
        d_model: hidden size (8 in this example)
        num_heads: number of sequential MTP steps (d)
        nhead: attention heads in each Transformer block
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_heads = num_heads

        # shared modules
        self.rmsnorm = RMSNorm(d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        # share weights between embed and unembed
        self.unembed.weight = self.embed.weight
        self.projections = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
            for _ in range(num_heads)
        ])

    def forward(self, token_ids: torch.LongTensor, init_hidden: torch.Tensor = None):
        """
        token_ids: (batch, seq_len) integer IDs of your input tokens 
        init_hidden: optional (batch, seq_len, d_model) base hidden states;

        """
        B, T = token_ids.shape
        device = token_ids.device
        embeds = self.embed(token_ids)

        # base hidden states
        if init_hidden is None:
            h0_seq = embeds
        else:
            h0_seq = init_hidden
        
        outputs = []
        max_i = T - self.num_heads - 1 # T is the input sequence length, num_heads is how much depth I am predicting into the future
        for i in range(0, max_i + 1):
            h_prev = h0_seq[:, i, :]
            # collect logits for all k at this i
            logits_k = []
            for k in range(self.num_heads):
                # future token embed at pos i + (k+1)
                future_pos = i + (k + 1)
                tok_embed = embeds[:, future_pos, :]
                h_norm = self.rmsnorm(h_prev)
                e_norm = self.rmsnorm(tok_embed)

                # concatenate
                merged = torch.cat([h_norm, e_norm], dim=-1)

                # project back to d_model
                proj = self.projections[k](merged)

                # Transformer block
                x = proj.unsqueeze(0)
                x = self.transformers[k](x)
                h_curr = x.squeeze(0)

                # unembed -> logits
                logits = self.unembed(h_curr)
                logits_k.append(logits)

                # chain hidden for next depth
                h_prev = h_curr
            
            # stack along depth axis
            logits_k = torch.stack(logits_k, dim=1)
            outputs.append(logits_k)




# loss function calculation between target tokens and predicted tokens
batch_size, seq_len, vocab_size = 1, 8, 5000
targets = torch.randint(0, vocab_size, (batch_size, seq_len))
print("targets.shape -", targets.shape) 
logits = model(tokens)
B, L, D, V = logits.shape
_, T = targets.shape
assert L == T-D

# double-loop loss:
loss = 0.0
for i in range(L):
    for k in range(D):
        logit_ik = logits[:, i, k, :]
        target_ik = targets[:, i + (k+1)] 
        loss += F.cross_entropy(logit_ik, target_ik)
loss = loss / (L*D)
print("MTP loss:", loss.item())