import torch
from torch import nn
import gym
from random import *

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
            nn.ReLU(),
            nn.Linear(16, 1, False),
            nn.Tanh()
        )
        # Freeze the learning of the newtork by deactivating gradient
        # Iterate over all parameters in the model and deactivate gradient
        for param in self.linear_network.parameters():
            param.requires_grad = False
            # takes in a module and applies the specified weight initialization

        #self.linear_network.apply(weights_init_uniform_rule)

    # Forward function through the network
    def forward(self, x):
        # Pass observation through forward function
        # Since the observation also contains other values, only take the value index at 0
        logits = self.linear_network(torch.tensor(x[0]))
        # Return normal python float value instead of tensor
        floatValue = logits.item()
        #print(floatValue)
        if (floatValue >= -0.33 and floatValue <= 0.33):
            floatValue = 0
        else:
            if (floatValue > 0.33):
                floatValue = 2
            else:
                floatValue = 1
        return floatValue

    # Set weigths and bias to given parameter from parent
    # By iterating over it like a 1D array
    def inherite_weights(self, weight_list, brother_weights, chance):
        new_weight = 0
        for layer_index, layer in enumerate(self.linear_network):
            if type(layer) == torch.nn.modules.linear.Linear:
                for input_index, input_weights in enumerate(layer.weight):
                    for x in range(self.linear_network[layer_index].in_features):
                        mutate = mutation_chance(chance)
                        if not mutate:
                            self.linear_network[layer_index].weight[input_index, x] = weight_list[new_weight]
                        else:
                            if flip_coin():
                                if flip_coin():
                                    mutated_weight = weight_list[new_weight] + weight_list[new_weight]*(randint(10,80)*0.01)
                                else:
                                    mutated_weight = weight_list[new_weight] - weight_list[new_weight]*(randint(10,80)*0.01)
                            else:
                                mutated_weight = weight_list[new_weight] + 0.1
                            self.linear_network[layer_index].weight[input_index, x] = mutated_weight
                        new_weight += 1


# Return either True or False randomly
def flip_coin():
    return bool(getrandbits(1))


def mutation_chance(chance):
    if randint(1, 100) > chance:
        return False
    else:
        return True


class Evolution:
    def __init__(self, max_gen, initial_size, survivor_size):
        self.max_generations = max_gen
        self.initial_population_size = initial_size
        self.survivor_population_size = survivor_size
        self.agent_list = []
        self.ranked_population = []
        self.mutation_prop = 90

    # create initial generation of random weigthed agents
    def make_gen_0(self):
        for amount in range(self.initial_population_size):
            self.agent_list.append(NeuralNetwork())

    # sort the list population by its rank
    # return sorted list of tuples (model,score)
    def rank_population(self, population_rank_list):
        self.ranked_population = sorted(population_rank_list, key=lambda tup: tup[1], reverse=True)
        self.agent_list.clear()
        for obj in self.ranked_population[:self.survivor_population_size]:
            self.agent_list.append(obj[0])

    # Take random weight values and return 2 new combination
    # Retrurns 1-D Array of future weights
    def cross_parents(self, parent1, parent2):
        # List of weights for the children
        inheritance_boolean_list_child_1 = []
        inheritance_boolean_list_child_2 = []
        # iterate over parents weights and randomly pick some weights
        for linear_net1, linear_net2 in zip(parent1.linear_network, parent2.linear_network):
            if type(linear_net1) == torch.nn.modules.linear.Linear and type(
                    linear_net2) == torch.nn.modules.linear.Linear:
                for node1, node2 in zip(linear_net1.weight, linear_net2.weight):
                    for weight1, weight2 in zip(node1, node2):
                        if flip_coin():
                            inheritance_boolean_list_child_1.append(weight1)
                            inheritance_boolean_list_child_2.append(weight2)
                        else:
                            inheritance_boolean_list_child_1.append(weight2)
                            inheritance_boolean_list_child_2.append(weight1)

        return (inheritance_boolean_list_child_1, inheritance_boolean_list_child_2)


p1 = Evolution(150, 200, 50)


# Get a better random distribution at the beginning
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / n ** 0.5
        m.weight.data.uniform_(-y, y)


score_list = []
loop_count = 0

# Main program loop
while True:
    if p1.ranked_population != []:
        for i in range (len(p1.ranked_population)-102):
            print(i,":", p1.ranked_population[i][1])
        print("end of list")
        loop_count += 1
    score_list = []
    p1.ranked_population = []
    # At first iteration set gen_0
    if (loop_count == 0):
        p1.make_gen_0()

    # Give every agent a round to play the game
    for index in range(len(p1.agent_list)):

        # Select the current rounds agent
        agent = p1.agent_list[index]
        # Set score achieved to 0
        maxScore = -11111111
        seed = 0
        # Run the game for every agent and get their scores
        # Render mode already given in the make process.
        env = gym.make('MountainCar-v0')

        # Number of steps you run the agent for
        num_steps = 200

        obs = env.reset(seed=0)
        input_space = obs
        output_space = env.action_space

        # Game loop for one game
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
                env.reset(seed=0)

        # Close the env
        env.close()
        score_list.append((agent, maxScore))

    p1.rank_population(score_list)
    if loop_count > 50 and p1.mutation_prop > 33:
        p1.mutation_prop -= 1
        print("mutation chance set to:", p1.mutation_prop)
    mutation_prop = p1.mutation_prop

    # stock up the population by crossing random survivors with each other until a certain threshold is met
    while p1.max_generations >= len(p1.agent_list):
        parent1 = randint(0, (p1.survivor_population_size - 1))
        parent2 = randint(0, (p1.survivor_population_size - 1))
        (child1_weights, child2_weights) = p1.cross_parents(p1.agent_list[parent1], p1.agent_list[parent2])
        instance1 = NeuralNetwork()
        instance2 = NeuralNetwork()
        instance1.inherite_weights(child1_weights, child2_weights, mutation_prop)
        instance2.inherite_weights(child2_weights, child1_weights, mutation_prop)
        p1.agent_list.append(instance1)
        p1.agent_list.append(instance2)
