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
num_experts = 3
top_k = 2 # this is the number of expert that the router will choose
n_embed = 8 # Embedding dimension

# Example multi-ead atttention output for a simple illustrative example, consider n_embed=32, context_length = 4
mh_output = torch.randn(1, 4, n_embed) # pretending the ouput from a multihead attention, the 1 is the batch size, 4 is the number of tokens

# maps from n_embed to num_expert
# The following will actually be my router matrix
topkgate_linear = nn.Linear(n_embed, num_experts) # nn.Linear(32, 4)

# generating score , Expert selector matrix
# the logits will likely be used later to pick the top-k experts for each token
logits = topkgate_linear(mh_output) # this will perform the matrix multiplication to get e1, e2, e3
print(logits)
# implementing the top k load balancing
top_k_logits, top_k_indices = logits.topk(top_k, dim=-1) # Get top-k experts
print(top_k_logits, top_k_indices)
# Now we replace the not selected values with negative infinity
# And then we will apply softmax
zeros = torch.full_like(logits, float('-inf')) # full_like clones a tensor and fill it will -infinity
sparse_logits = zeros.scatter(-1, top_k_indices, top_k_logits)
print(sparse_logits)
gating_ouput = F.softmax(sparse_logits, dim=-1)
print(gating_ouput)

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
num_experts = 3
top_k = 2
n_embd = 8
mh_output = torch.randn(1, 4, n_embed) 
top_k_gate = TopkRouter(n_embed, num_experts, top_k)
gating_ouput, indices = top_k_gate(mh_output)
print(gating_ouput.shape, gating_ouput, indices)

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
    
num_experts = 3
top_k = 2
n_embd = 8
mh_output = torch.randn(1, 4, n_embed) 
top_k_gate = NoisyTopkRouter(n_embed, num_experts, top_k)
gating_ouput, indices = top_k_gate(mh_output)
print(gating_ouput.shape, gating_ouput, indices)

