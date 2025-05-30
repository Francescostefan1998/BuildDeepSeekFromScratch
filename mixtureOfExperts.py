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
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential( # hold the actual layers, nn.Sequential stacks layers in order
            nn.Linear(n_embd, 4* n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )