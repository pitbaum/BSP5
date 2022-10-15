import torch
from torch import nn
from torchsummary import summary

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Agent_newtork:
    """
    #hardcoded topology
    model = torch.nn.Sequential(
        nn.Linear(3, 3), #3 inputs
        nn.LeakyReLU(), #leaky relu for also negative values
        nn.Linear(3, 1) #1 output, velocity
    )
    """

    #Initialize Network with topology (model) and parameters (tensor) passed 
    def __init__(self, model, parameters):
        #create model
        self.model = model
        #set model gradient to false
        deactivate_gradient(parameters)
        #put states in a dict
        state_dict = model.state_dict()
        #change the weights to the parameter given
        for name, param in state_dict.items():
            # Don't update if this is not a weight.
            if not "weight" in name:
                continue
            # Update the parameter.
            param.copy_(parameters)

    #freeze the learning of the newtork by deactivating gradient
    def deactivate_gradient():
        #iterate over all parameters in the model and deactivate gradient
        for param in self.model.parameters():
            param.requires_grad = False

class Root_network:
    def __init__(self):
        #hardcoded topology
        self.model = torch.nn.Sequential(
            nn.Linear(3, 3), #3 input layers
            nn.LeakyReLU(), #leaky relu for also negative values
            nn.Linear(3, 1) #1 output, velocity
        )
        #freeze the learning of the newtork by deactivating gradient
        #iterate over all parameters in the model and deactivate gradient
        for param in self.model.parameters():
            param.requires_grad = False