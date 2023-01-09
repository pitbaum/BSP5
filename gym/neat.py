import gym
import random
import math

#Please read the README instructions about this file before attempting to launch it.
print("Start of program")

def getRandomWeight():
    return random.uniform(-2,2)

def getTanh(x):
    return math.tan(x)

# A neural network class shape that is extendible
# The initial shape of the network is a minimal representation of the parents with 1 hidden node
# The maximum layers are input, 3 hidden layers, 1 output layer
# TanH function applied to the end result of the output layer, could add a relu to the 2nd layer result in forward?
class FeedForwardNetwork:
    def __init__(self):
        self.node_id_count = 3
        # Set the first input node id 0 to be in the 1st layer index 0
        # No input nodes linked []
        self.input_x = Node(0, [], 0)
        # Set the second input node id 1 to be in the 1st layer index 0
        # No input nodes linked []
        self.input_y = Node(1, [], 0)
        # Create a node between in an output of the function
        # node id 2, input nodes [(x.id,weight),(y.id,weight)], layer index 1
        self.connector_node = Node(2, [(self.input_x.id, getRandomWeight()), (self.input_y.id, getRandomWeight())], 2)
        # Create the output node of the model
        # node id 3, input node [(connector.id,weight)], layer index 4
        self.output_node = Node(3, [(self.connector_node.id, getRandomWeight())], 4)
        # Create the input layer dict of nodes
        # Key node id, value = node
        self.inputLayer = {self.input_x.id: self.input_x, self.input_y.id: self.input_y}
        # create the empty layer dict for layer id 1
        self.layer1 = {}
        # Create the layer with id 2 and connector node inside
        self.layer2 = {self.connector_node.id: self.connector_node}
        # Create the empty layer dict with id 3
        self.layer3 = {}
        # Create the output layer with the output node inside
        self.outputLayer = {self.output_node.id: self.output_node}
        # Put all the layers in order into a list of layers
        self.layerList = [self.inputLayer, self.layer1, self.layer2, self.layer3, self.outputLayer]
        self.allNodesDict = {self.input_x.id:self.input_x,self.input_y.id:self.input_y,self.connector_node.id:self.connector_node,self.output_node.id:self.output_node}
        self.normalizedNodesDict = {}

    # Node n.value = sum of nodes.value in its input list times their weight in the list
    # Input values x and y are the floats from the environment passed to the forward function
    def forward_pass(self, x, y):
        # Initialize the 2 input nodes value with the passed parameters x and y
        self.input_x.value = x
        self.input_y.value = y
        # For every layer, iterate over all nodes in it and calculate their value in order of layer id
        # (From input layer to output layer)
        for layer in self.layerList:
            # Dont iterate over empty layer dicts since they dont have nodes
            # If dict: returns false if dict is only {}
            if layer:
                # Iterate over the dict of nodes in the layer and do some operation for each node in it
                for current_node in layer.values():
                    # If the in connections list of the current node is not empty
                    # Else dont do any operation, since the node doesnt take any inputs
                    if current_node.con_in:
                        # Iterate over the list of in connections to the node
                        # in_tuple is the tuple representing the node connection: (connected_node, weight_of_connection)
                        for in_tuple in current_node.con_in:
                            # Add the value of this connection to the value of the node
                            # Add to intermediate result: connection node value * weight of connection
                            node_id = None
                            weight = None
                            isWeight = False
                            for i in in_tuple:
                                if isWeight:
                                    weight = i
                                else:
                                    node_id = self.allNodesDict.get(i).value
                                    isWeight = True
                            if node_id != None and isWeight != None:
                                current_node.addToValue(node_id, weight)
        # Safe the result of the forward pass
        result = self.output_node.value
        # Set all node values to 0 again, the value should not be stored in between runs.
        for node in self.allNodesDict.values():
            node.setValueToZero()
        # Return value of output node
        # Put the value through tanh function
        floatValue = getTanh(result)
        # Evaluate to 0,1 or 2 as a value for the action space
        if (floatValue >= -0.33 and floatValue <= 0.33):
            floatValue = 0
        else:
            if (floatValue > 0.33):
                floatValue = 2
            else:
                floatValue = 1
        return floatValue

    # Choose a random layer and a random node in it
    # Take a random in node from it and replace it
    # Put this node in the layer between the in and out node set in node to original weight, out node to 1
    # If there is no layer in between the nodes, make a new layer for it
    def mutate_insert_node(self):
        # Choose a random layer index of the layers receiving input
        chosenLayer = random.randint(2, 4)
        # If the layer has no nodes, dont do anything. End of progrem
        # If the layer is non empty:
        if self.layerList[chosenLayer]:
            # Choose a random node from the layer
            outNodeIndex = random.randint(0, len(self.layerList[chosenLayer])-1)
            outNodeKey = list(self.layerList[chosenLayer])[outNodeIndex]
            outNode = self.layerList[chosenLayer].get(outNodeKey)
            # Choose a random node from its input nodes
            inNodeIndex = random.randint(0, len(outNode.con_in)-1)
            inNodeTuple = outNode.con_in[inNodeIndex]
            inNode = self.allNodesDict.get(inNodeTuple[0])
            # Get the connection weight
            weight = inNodeTuple[1]
            # Increase the max id of nodes in the program
            self.node_id_count += 1
            # Create a new node with the new max id, the (inNode.id,weight) of the in connection and the layer under the out node connection
            newNode = Node(self.node_id_count,[(inNode.id,weight)],chosenLayer-1)
            # Add the new node to the all nodes dict, that contains all existing nodes
            self.allNodesDict.update({self.node_id_count:newNode})
            # Change the inNode connection of the outNode at the selected place to the new (node.id,weight) with weight 1
            outNode.con_in[inNodeIndex] = (self.node_id_count,1)

    # Since the program is restircted to layers, it needs to be implemented that you can also add a node
    # Without putting it between two already existing nodes
    def mutate_add_node(self):
        # Put a random new node in either layer 1,2 or 3
        chosenLayer = random.randint(1,3)
        # If the layer under it is empty, take the layer under that one
        if self.layerList[chosenLayer-1] == {}:
            inLayer = chosenLayer-2
        else:
            inLayer = chosenLayer-1
        # Choose the node that will be inputted and get its id
        inNodeIndex = random.randint(0,len(self.layerList[inLayer])-1)
        inNodeKey = list(self.layerList[inLayer])[inNodeIndex]
        # If the layer above it is empty, take the layer above that for outputting
        if self.layerList[chosenLayer+1] == {}:
            outLayer = chosenLayer+2
        else:
            outLayer = chosenLayer+1
        # Choose the output node and get its id
        outNodeIndex = random.randint(0,len(self.layerList[outLayer])-1)
        outNodeKey = list(self.layerList[outLayer])[outNodeIndex]
        # Create a new node with a new id at the given layer, with in node and a random weight
        self.node_id_count += 1
        newNode = Node(self.node_id_count,[(inNodeKey,getRandomWeight())],chosenLayer)
        # Put the new node in the layer and in the nodes dict
        self.layerList[chosenLayer].update({newNode.id:newNode})
        self.allNodesDict.update({newNode.id:newNode})
        # Put the new nodes connection in the in connections of the selected output node with random weight
        self.allNodesDict.get(outNodeKey).con_in.append((newNode.id,getRandomWeight()))


    # Choose a random layer and a random node in it
    # Choose a random node from the layer + 1
    # Add a connection from that node to the other and add a random weight to it
    def mutate_add_connection(self):
        # choose a random layer index
        chosenLayer = random.randint(1, 4)
        if self.layerList[chosenLayer]:
            inNodeIndex = random.randint(0, len(self.layerList[chosenLayer]) - 1)
            inNodeKey = list(self.layerList[chosenLayer])[inNodeIndex]
            outNodeIndex = random.randint(0,len(self.layerList[chosenLayer-1])-1)
            outNodeKey = list(self.layerList[chosenLayer-1])[outNodeIndex]
            inNode = self.allNodesDict.get(inNodeKey)
            exists = False
            for id in inNode.con_in:
                isId = True
                for value in id:
                    if isId:
                        if value == id:
                            exists = True
                        isId = False
                    else:
                        isId = True
            if not exists:
                inNode.con_in.append((outNodeKey,getRandomWeight()))

    # Add a node without any input connections into the layer passed as parameter
    # Used as help function for the cross parents function
    def add_node(self, chosenLayer):
        self.node_id_count += 1
        newNode = Node(self.node_id_count, [], chosenLayer)
        # Put the new node in the layer and in the nodes dict
        self.layerList[chosenLayer].update({newNode.id: newNode})
        self.allNodesDict.update({newNode.id: newNode})

    # Iterate over all layers and change the node normalized ids to be strictly ascending by index
    # Change the values for the  access of the normalized all nodes dict
    def normalize_ids(self):
        current_id = 0
        new_normalized_dict = {}
        for layer in self.layerList:
            for nodeKey in layer:
                node = self.allNodesDict.get(nodeKey)
                node.normalized_id = current_id
                new_normalized_dict.update({node.normalized_id:node})
        self.normalizedNodesDict = new_normalized_dict

# A single node
# The nodes unique id number to identify it
# The list of input connections to the node
# Its current layer number spanning from 0-4
# The nodes value
class Node:
    def __init__(self, id, in_list, layer_id):
        # id of the current node
        self.id = id
        # Normalized id
        self.normalized_id = -1
        # tuple (in node id, weight)
        # Make into dictionary from list
        self.con_in = in_list
        # For the cross parent function and get norm con dict fucntion, use a dictionary
        self.con_in_dict = {}
        # layer number
        self.layer_id = layer_id
        # value of the node
        self.value = 0

    def getInNodeDict(self):
        for tuple in self.con_in:
            self.con_in_dict.update({tuple[0]:tuple[1]})

    def makeInNodeList(self):
        self.con_in = []
        for key,value in zip(self.con_in_dict):
            self.con_in.append((key,value))
    def get_norm_con_dict(self, allNodesDict):
        new_dict = {}
        inDict = self.getInNodeDict()
        for key in inDict:
            weight = inDict.get(key)
            new_key = allNodesDict.get(key).normalized_id
            new_dict.update({new_key:weight})
        return new_dict
    # Function to sum the current value with the connection of the connected node times its weight to it
    def addToValue(self, value, weight):
        self.value += value * weight

    def setValueToZero(self):
        self.value = 0

""""
# Create new Neural net instance
testNet = FeedForwardNetwork()
x = 0.5
y = 0.5
testNet.__init__()
print(testNet.forward_pass(x, y))
print(testNet.allNodesDict)
testNet.add_node()
print(testNet.allNodesDict)
print(testNet.forward_pass(x,y))
"""

# Return either True or False randomly
def flip_coin():
    return bool(random.getrandbits(1))

def mutate(weight,chance):
    mutate = mutation_chance(chance)
    if not mutate:
        return weight
    else:
        if flip_coin():
            if flip_coin():
                return weight + weight * (random.randint(10, 80) * 0.01)
            else:
                return weight - weight * (random.randint(10, 80) * 0.01)
        else:
            return (weight + 0.1)

def mutation_chance(chance):
    if random.randint(1, 100) > chance:
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
    # All with the initial starting shape of 2,1,tanh(1)
    def make_gen_0(self):
        for amount in range(self.initial_population_size):
            self.agent_list.append(FeedForwardNetwork())

    # sort the list population by its rank
    # return sorted list of tuples (model,score)
    def rank_population(self, population_rank_list):
        self.ranked_population = sorted(population_rank_list, key=lambda tup: tup[1], reverse=True)
        self.agent_list.clear()
        for obj in self.ranked_population[:self.survivor_population_size]:
            self.agent_list.append(obj[0])

    # Take the 2 parents and return 2 children with mutated inherited weights and max topology of (parent1,parent2)
    # Returns (child1,child2)
    def cross_parents(self, parent1, parent2, chance):
        child1 = FeedForwardNetwork()
        child2 = FeedForwardNetwork()

        parent1.normalize_ids()
        parent2.normalize_ids()

        currentLayer = -1
        currentNormNodeId = 0
        for layer in parent1.layerList:
            currentLayer += 1
            # As long as there are still some nodes in the layer of one parent that you didnt iterate through
            while currentNormNodeId <= len(parent1.layerList[currentLayer]) or currentNormNodeId <= len(parent2.layerList[currentLayer]):
                child1.add_node(currentLayer)
                child2.add_node(currentLayer)
                currentNodeId1 = None
                currentNodeId2 = None
                if currentNormNodeId <= len(parent1.layerList[currentLayer]):
                    currentNodeId1 = parent1.normalizedNodesDict.get(currentNormNodeId).id
                if currentNormNodeId <= len(parent1.layerList[currentLayer]):
                    currentNodeId2 = parent1.normalizedNodesDict.get(currentNormNodeId).id
                # If this node exists for parent1 and parent2
                if parent1.layerList[currentLayer].get(currentNodeId1) and parent2.layerList[currentLayer].get(currentNodeId2):
                    node1 = parent1.layerList[currentLayer].get(currentNodeId1)
                    node2 = parent2.layerList[currentLayer].get(currentNodeId2)
                    # Create the dict of normed ids with in weights
                    normInDict1 = node1.get_norm_con_dict(node1.allNodesDict)
                    normInDict2 = node2.get_norm_con_dict(node2.allNodesDict)
                    # Go over all connections in parent1, add all necessary/possible mutual or parent1 connections and delete them from the dict(s)
                    # mutate weights before putting them in the childrens con in dicts if applicable
                    for key1 in normInDict1:
                        # If a connection is mutual, give child1 one weight and child2 the other
                        if normInDict1.get(key1) and normInDict2.get(key1):
                            if flip_coin():
                                child1.allNodesDict.get(key1).con_in.update({key1:mutate(normInDict1.get(key1),chance)})
                                child2.allNodesDict.get(key1).con_in.update({key1:mutate(normInDict2.get(key1),chance)})
                            else:
                                child1.allNodesDict.get(key1).con_in.update({key1:mutate(normInDict2.get(key1),chance)})
                                child2.allNodesDict.get(key1).con_in.update({key1:mutate(normInDict1.get(key1),chance)})
                            # Delete the reference in the norm dict to not iterate over it again
                            normInDict1.pop(key1,None)
                            normInDict2.pop(key1,None)
                        # If only parent1 has the connection, give both children the same connection
                        elif normInDict1.get(key1):
                            child1.allNodesDict.get(key1).con_in.update({key1:mutate(normInDict1.get(key1),chance)})
                            child2.allNodesDict.get(key1).con_in.update({key1:mutate(normInDict1.get(key1),chance)})
                            normInDict1.pop(key1,None)
                    # Go over parent2 with the remaining keys and add the non mutual one's that remain
                    for key2 in normInDict2:
                        child1.allNodesDict.get(key2).con_in.update({key2:mutate(normInDict2.get(key2),chance)})
                        child2.allNodesDict.get(key2).con_in.update({key2:mutate(normInDict2.get(key2),chance)})
                        normInDict2.pop(key2,None)
                    # Both dicts should be empty by now since all connections were made and deleted
                    if normInDict1 == {} and normInDict2 == {}:
                        print("Success of joining the genes")
        return (child1,child2)


p1 = Evolution(150, 200, 50)
score_list = []
loop_count = 0

# Main program loop
while True:
    if p1.ranked_population != []:
        for i in range(len(p1.ranked_population) - 102):
            print(i, ":", p1.ranked_population[i][1])
        print("end of list")
        loop_count += 1
        if p1.ranked_population[0][1] >= 0.5:
            break
    score_list = []
    p1.ranked_population = []
    # At first iteration set gen_0
    if (loop_count == 0):
        p1.make_gen_0()

    # Run the game for every agent and get their scores
    # Render mode already given in the make process.
    env = gym.make('MountainCar-v0')
    # Give every agent a round to play the game
    for index in range(len(p1.agent_list)):

        # Select the current rounds agent
        agent = p1.agent_list[index]
        # Set score achieved to 0
        maxScore = -11111111
        seed = 0

        # Number of steps you run the agent for
        num_steps = 200

        obs = env.reset(seed=0)
        input_space = obs
        output_space = env.action_space

        # Game loop for one game
        for step in range(num_steps):
            # take action
            # Pass the observation space, x coord and velocity to the network input
            action = agent.forward_pass(obs[0][0],obs[0][1])
            # apply the action
            obs = env.step(action)
            # Update the furthest the car came on the x axis
            if maxScore < obs[0][0]:
                maxScore = obs[0][0]
            # If the epsiode is up, then start another one
            if obs[2]:
                env.reset(seed=0)

        score_list.append((agent, maxScore))

    # Close the env
    env.close()

    p1.rank_population(score_list)
    if loop_count > 50 and p1.mutation_prop > 33:
        p1.mutation_prop -= 1
        print("mutation chance set to:", p1.mutation_prop)
    mutation_prop = p1.mutation_prop

    # stock up the population by crossing random survivors with each other until a certain threshold is met
    while p1.max_generations >= len(p1.agent_list):
        parent1 = random.randint(0, (p1.survivor_population_size - 1))
        parent2 = random.randint(0, (p1.survivor_population_size - 1))
        (child1, child2) = p1.cross_parents(p1.agent_list[parent1], p1.agent_list[parent2],mutation_prop)
        p1.agent_list.append(child1)
        p1.agent_list.append(child2)

""" Show the end result of the best agent that can reach the goal state"""
final_agent = p1.ranked_population[0][0]
print("came here")
env = gym.make('MountainCar-v0', render_mode="human")
num_steps = 200
obs = env.reset(seed=0)
input_space = obs
output_space = env.action_space

# Game loop for one game
for step in range(num_steps):
    # take action
    # action = agent.act(obs)
    action = final_agent(obs)
    # apply the action
    obs = env.step(action)