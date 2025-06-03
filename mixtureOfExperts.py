import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)
# Download the tiny shakespeare dataset
# !wget https://raw.githubusercontent.com/AviSoori1x/makeMoE/main/input.txt
# import requests

# url = "https://raw.githubusercontent.com/AviSoori1x/makeMoE/main/input.txt"
# response = requests.get(url)

# with open("input.txt", "w", encoding="utf-8") as f:
#     f.write(response.text)
# with open("input.txt", "r", encoding="utf-8") as f:
#     text = f.read()
# chars = sorted(list(set(text)))
# vocab_size = len(chars)

   
#Epert module
class Expert(nn.Module):
    # An MLP is a simple liear layer followed by a non-linearity i.e. each Expert
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential( # hold the actual layers, nn.Sequential stacks layers in order, like a pipeline
            nn.Linear(n_embd, 4* n_embd), # expands dimensionality by 4
            nn.ReLU(), # adds non-linearity
            nn.Linear(4 * n_embd, n_embd), # projects back to the original size
            nn.Dropout(dropout), # regularization to avoid overfitting
        )

    def forward(self, x):
        return self.net(x)
    
# Understanding how gating works
# num_experts = 3
# top_k = 2 # this is the number of expert that the router will choose
# n_embed = 8 # Embedding dimension

# # Example multi-ead atttention output for a simple illustrative example, consider n_embed=32, context_length = 4
# mh_output = torch.randn(1, 4, n_embed) # pretending the ouput from a multihead attention, the 1 is the batch size, 4 is the number of tokens

# # maps from n_embed to num_expert
# # The following will actually be my router matrix
# topkgate_linear = nn.Linear(n_embed, num_experts) # nn.Linear(32, 4)

# # generating score , Expert selector matrix
# # the logits will likely be used later to pick the top-k experts for each token
# logits = topkgate_linear(mh_output) # this will perform the matrix multiplication to get e1, e2, e3
# print(logits)
# # implementing the top k load balancing
# top_k_logits, top_k_indices = logits.topk(top_k, dim=-1) # Get top-k experts
# print(top_k_logits, top_k_indices)
# # Now we replace the not selected values with negative infinity
# # And then we will apply softmax
# zeros = torch.full_like(logits, float('-inf')) # full_like clones a tensor and fill it will -infinity
# sparse_logits = zeros.scatter(-1, top_k_indices, top_k_logits)
# print(sparse_logits)
# gating_ouput = F.softmax(sparse_logits, dim=-1)
# print(gating_ouput)

class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_ouput):
        # is the output tensor from the multihead self attention block
        logits = self.linear(mh_ouput)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices
    
# Testing
# num_experts = 3
# top_k = 2
# n_embd = 8
# mh_output = torch.randn(1, 4, n_embed) 
# top_k_gate = TopkRouter(n_embed, num_experts, top_k)
# gating_ouput, indices = top_k_gate(mh_output)
# print(gating_ouput.shape, gating_ouput, indices)

# Create some noise

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        # layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)
    
    def forward(self, mh_ouput):
        # mh_output is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_ouput)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices
    
# num_experts = 3
# top_k = 2
# n_embd = 8
# mh_output = torch.randn(1, 4, n_embed) 
# top_k_gate = NoisyTopkRouter(n_embed, num_experts, top_k)
# gating_ouput, indices = top_k_gate(mh_output)
# print(gating_ouput.shape, gating_ouput, indices)

# sparse mixture of experct module

class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed, 0.1) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_ouput = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in the top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input) # here happen the real matrix multiplication amongst experts

                # Extract and apply gating scores
                gating_scores = flat_gating_ouput[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output*gating_scores

                # Update the final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output
    
num_experts = 3
top_k = 2
n_embed = 8
dropout=0.1
mh_output = torch.randn(1, 4, n_embed) 
sparse_moe = SparseMoE(n_embed, num_experts, top_k)
final_output = sparse_moe(mh_output)
print("Shape of the final output:", final_output.shape)
print(final_output)

class Head(nn.Module):
    # one head of self attention

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute attention score
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out
    
# Multi-Headed Self Attention
class MultiHeadAttention(nn.Module):

    # multiple heads of self-attention in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

# coding the transformer block
class Block(nn.Module):
    # Mixture of Expert transformer block
    def __init__(self, n_embed, n_head, num_expert, top_k):
        # n_embed: embedding dimension, h_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # self attention
        x = x + self.sa(self.ln1(x))
        # sparse mixture of experts
        x = x + self.smoe(self.ln2(x))
        return x

block_size = 128  # or whatever sequence length you want
vocab_size = 65   # placeholder; set this after processing the dataset
n_embed = 64      # embedding dimension
n_head = 4        # number of attention heads
num_experts = 3
top_k = 2
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# now we code the entire llmodel architecture
class SparseMoELanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head, num_experts=num_experts, top_k=top_k) for _ in range()])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x) # entire transformer block
        x = self.ln_f(x) # for the output, this is the layer normalization
        logits = self.lm_head(x) # from the embedding dimension to the vocabolary space

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def genearate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) 
            # sample from distribution
            idx_next = torch.mulitnomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next),dim=1)
        return idx