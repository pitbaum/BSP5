import torch
from torch import nn
from torchsummary import summary

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        # No flatten necessary since array input already [x,y]
        # Linear network with Tanh, 2 input 1 output
        self.linear_network = nn.Sequential(
            nn.Linear(2, 16),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        # freeze the learning of the newtork by deactivating gradient
        # iterate over all parameters in the model and deactivate gradient
        for param in self.linear_network.parameters():
            param.requires_grad = False
        #The amount of infeatures of the layers. If not applicable enter 0
        self.in_features = [2,16,0,16,0]

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_network(x)
        return logits

    def printShit(self):
        for count, item in enumerate(self.linear_network):
            if type(item) == torch.nn.modules.linear.Linear:
                for count2,item2 in enumerate(item.weight):
                    print(count,count2,item2)
                    for x in range(self.in_features[count]):
                        self.linear_network[count].weight[count2,x] = 1

model = NeuralNetwork()
model.printShit()