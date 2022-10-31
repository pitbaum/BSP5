import torch
from torch import nn
from torchsummary import summary

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # No flatten necessary since array input already [x,y]
        # Linear network with Tanh, 2 input 1 output
        # No grad and no bias
        self.linear_network = nn.Sequential(
            nn.Linear(2, 16, False),
            nn.Linear(16, 16, False),
            nn.ReLU(),
            nn.Linear(16, 1, False),
            nn.ReLU()
        )
        # Freeze the learning of the newtork by deactivating gradient
        # Iterate over all parameters in the model and deactivate gradient
        for param in self.linear_network.parameters():
            param.requires_grad = False

    # Forward function through the network
    def forward(self, x):
        # Pass observation through forward function
        # Since the observation also contains other values, only take the value index at 0
        logits = torch.clamp(self.linear_network(torch.tensor(x[0])), min=0, max=2)
        # Return normal python float value instead of tensor
        floatValue = logits.item()

        if (floatValue >= 0 and floatValue < 0.75):
            floatValue = 0
        else:
            if (floatValue >= 0.75 and floatValue < 1.25):
                floatValue = 1
            else:
                floatValue = 2
        return floatValue

    # Set weigths and bias to given parameter from parent
    # By iterating over it like a 1D array
    def inherite_weights(self, weight_list):
        new_weight = 0
        for layer_index, layer in enumerate(self.linear_network):
            if type(layer) == torch.nn.modules.linear.Linear:
                for input_index, input_weights in enumerate(layer.weight):
                    for x in range(self.in_features[layer_index]):
                        self.linear_network[layer_index].weight[input_index, x] = weight_list[new_weight]
                        new_weight += 1

p1 = NeuralNetwork()


print("2")
for i in p1.linear_network:
    print