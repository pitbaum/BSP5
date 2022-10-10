import torch
from torch import nn
from torchsummary import summary

model = torch.nn.Sequential(
        nn.Linear(3, 3), #3 input layers
)

def init_model(parameters):
    for old, new in zip(model.parameters(), parameters):
        old.requires_grad = False #freeze the learning of the newtork by deactivating steepest decent gradient call for every parameter (with torch.no_grad():)?
        old = new # set the weight to the new value


state_dict = model.state_dict()

for name, param in state_dict.items():
    # Don't update if this is not a weight.
    if not "weight" in name:
        continue
    print(param)

    # Transform the parameter as required.
    transformed_param = param * -1

    print(transformed_param)
    # Update the parameter.
    param.copy_(transformed_param)

