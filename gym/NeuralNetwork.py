import random
from random import getrandbits
import torch
from torch import nn
import gym

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device", device)


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


# Return either True or False randomly
def flip_coin():
    return bool(getrandbits(1))


class Evolution:
    def __init__(self, max_gen, initial_size, survivor_size):
        self.max_generations = max_gen
        self.initial_population_size = initial_size
        self.survivor_population_size = survivor_size
        self.agent_list = []
        self.ranked_population = []

    # create initial generation of random weigthed agents
    def make_gen_0(self):
        for amount in range(self.initial_population_size):
            self.agent_list.append(NeuralNetwork())

    # sort the list population by its rank
    # return sorted list of tuples (model,score)
    def rank_population(self, population_rank_list):
        self.ranked_population = (sorted(population_rank_list, key=lambda x: x[1]))
        self.agent_list.clear()
        for obj in self.ranked_population[:self.survivor_population_size]:
            self.agent_list.append(obj[0])


    # Take random weight values and return 2 new combination
    # Retrurns 1-D Array of future weights
    def cross_parents(self, parent1, parent2):
        # List of weights for the children
        inheritance_boolean_list_child_1 = []
        inheritance_boolean_list_child_2 = []
        print("this is a layer that throws an error", parent1.linear_network)
        # iterate over parents weights and randomly pick some weights
        for linear_net1, linear_net2 in parent1.modules(), parent2.modules():
            if type(linear_net1) == torch.nn.modules.linear.Linear and type(
                    linear_net1) == torch.nn.modules.linear.Linear:
                for node1, node2 in linear_net1.weight, linear_net2.weight:
                    for weight1, weight2 in node1, node2:
                        if flip_coin():
                            inheritance_boolean_list_child_1.append(weight1)
                            inheritance_boolean_list_child_2.append(weight2)
                        else:
                            inheritance_boolean_list_child_1.append(weight2)
                            inheritance_boolean_list_child_2.append(weight1)

        return (inheritance_boolean_list_child_1, inheritance_boolean_list_child_2)


p1 = Evolution(100, 100, 50)
score_list = []

for index in range(p1.max_generations):
    # At first iteration set gen_0
    if (index == 0):
        p1.make_gen_0()

    # Select the current rounds agent
    agent = p1.agent_list[index]
    # Set score achieved to 0
    maxScore = -11111111

    # Run the game for every agent and get their scores
    # Render mode already given in the make process.
    env = gym.make('MountainCar-v0')

    # Number of steps you run the agent for
    num_steps = 1500

    obs = env.reset()
    input_space = obs
    output_space = env.action_space

    for step in range(num_steps):
        # take action
        # action = agent.act(obs)
        action = agent(obs)
        # apply the action
        obs = env.step(action)
        # Update the furthest the car came on the x axis
        if maxScore < obs[0][0]:
            maxScore = obs[0][0]
        # If the epsiode is up, then start another one
        if obs[2]:
            env.reset()

    # Close the env
    env.close()
    score_list.append((agent, maxScore))

p1.rank_population(score_list)

# stock up the population by crossing random survivors with each other
while p1.initial_population_size >= len(p1.agent_list):
    parent1 = random.randint(0, (p1.survivor_population_size-1))
    parent2 = random.randint(0, (p1.survivor_population_size-1))
    print(p1.agent_list[parent1])
    (child1_weights, child2_weights) = p1.cross_parents(p1.agent_list[parent1], p1.agent_list[parent2])
    p1.agent_list.append(NeuralNetwork().inherite_weights(child1_weights))
    p1.agent_list.append(NeuralNetwork().inherite_weights(child2_weights))

print(len(p1.agent_list))
