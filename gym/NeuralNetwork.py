import os
from random import getrandbits
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device", device)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        # No flatten necessary since array input already [x,y]
        # Linear network with Tanh, 2 input 1 output
        self.linear_network = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        # freeze the learning of the newtork by deactivating gradient
        # iterate over all parameters in the model and deactivate gradient
        for param in self.linear_network.parameters():
            param.requires_grad = False

    # forward function through the network
    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_network(x)
        return logits

    # Set weigths and bias to given parameter from parent
    def inherite_weights_bias(self, weight_bias_list):
        for linear_net in self.linear_network:
            # If module function is a linear layer
            if type(linear_net) == torch.nn.modules.linear.Linear:
                print("Module Number changed and we are in deeper")
                for node in linear_net.weight:
                    for weight in node:
                        print(weight)
                # TODO: Enumerate over weight bias list and set the values at the right place


# Take random weight values and return 2 new combination
# Retrurns 1-D Array of future weights
def cross_parents(parent1, parent2):
    # List of weights for the children
    inheritance_boolean_list_child_1 = []
    inheritance_boolean_list_child_2 = []
    # iterate over parents weights and randomly pick some weights
    for linear_net1, linear_net2 in parent1, parent2:
        if type(linear_net1) == torch.nn.modules.linear.Linear and type(linear_net1) == torch.nn.modules.linear.Linear:
            for node1, node2 in linear_net1.weight, linear_net2:
                for weight1, weight2 in node1, node2:
                    if flip_coin():
                        inheritance_boolean_list_child_1.append(weight1)
                        inheritance_boolean_list_child_2.append(weight2)
                    else:
                        inheritance_boolean_list_child_1.append(weight2)
                        inheritance_boolean_list_child_2.append(weight1)

    return (inheritance_boolean_list_child_1, inheritance_boolean_list_child_2)

# Return either True or False randomly
def flip_coin():
    return bool(getrandbits(1))

model = NeuralNetwork().to(device)
print(model)

# Random created input value
X = torch.rand(2, device=device)
print(X)
logits = model(X)
print(logits)

# Check if gradient is disabled
for param in model.linear_network.parameters():
    print(param.requires_grad)

# print("model state dict: ", model.state_dict())

model.inherite_weights_bias("a")
