import torch
from torch import nn
from torchsummary import summary
import gym
import matplotlib.pyplot as plt 
import time


#Render mode already given in the make process.
env = gym.make('MountainCar-v0', render_mode = "human")

# Number of steps you run the agent for 
num_steps = 1500

obs = env.reset()
input_space = obs
output_space = env.action_space

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_space, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_space),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

for step in range(num_steps):
    # take random action
    action = model(obs)
    #action = env.action_space.sample()
    # apply the action
    obs, reward, done, info, empty = env.step(action)
    # Wait a bit before the next frame unless you want to see a crazy fast video
    time.sleep(0.001)
    # If the epsiode is up, then start another one
    if done:
        env.reset()

# Close the env
env.close()