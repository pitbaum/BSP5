import os
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device", device)

class NeuralNetwork(nn.Module):
    def __int__(self):
        super(NeuralNetwork, self).__int__()
        #self.flatten = nn.Flatten()
        #no need for flatten? Since already 1-D array input?
        self.linear_tanh = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_tanh(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

X = torch.rand(2, 1, device = device)
print(X)
logits = model(X)
print(logits)