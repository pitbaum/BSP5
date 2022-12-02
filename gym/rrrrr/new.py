# Store the current max node id globally to make incremented ids for new nodes
import random

node_id_count = 3

def getRandomWeight():
    return random.uniform(-2,2)

class FeedForwardNetwork:
    def __init__(self):
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
        self.inputLayer = {self.input_x.id: self.input_x, self.input_y: self.input_y}
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
        return result


"""
    # Choose a random layer and a random node in it
    # Take a random in node from it and replace it
    # Put this node in the layer between the in and out node set in node to original weight, out node to 1
    # If there is no layer in between the nodes, make a new layer for it
    def mutate_add_node(self):
        chosenLayer = np.random.randint(1, len(self.layerList) - 1)
        outNode = np.random.randint(0, len(self.layerList[chosenLayer]))
        inNode = np.random.randint(0, len(outNode.con_in))
        # if inNode.layer_id+1 < outNode.layer_id:

    # Choose a random layer and a random node in it
    # Choose a random node from the layer + 1
    # Add a connection from that node to the other and add a random weight to it
    # def mutate_add_connection(self):
"""


# A single node
# The nodes unique id number to identify it
# The list of input connections to the node
# Its current layer number spanning from 0-4
# The nodes value
class Node:
    def __init__(self, id, in_list, layer_id):
        # id of the current node
        self.id = id
        # tuple (in node id, weight)
        self.con_in = in_list
        # layer number
        self.layer_id = layer_id
        # value of the node
        self.value = 0

    # Function to sum the current value with the connection of the connected node times its weight to it
    def addToValue(self, value, weight):
        self.value += value * weight

    def setValueToZero(self):
        self.value = 0
# Create new Neural net instance
testNet = FeedForwardNetwork()
x = 0.5
y = 0.5
testNet.__init__()
print(testNet.forward_pass(x, y))
