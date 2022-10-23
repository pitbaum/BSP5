import torch
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device", device)
print(torch.rand(2, device=device))