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

   
