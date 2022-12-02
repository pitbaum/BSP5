import numpy as np

class FeedForwardNetwork:
    def __int__(self):
        np.random.seed(0)
        self.global_node_id = 3
        self.x = node(0,[],0)
        self.y = node(1, [],0)
        self.connector_node = node(2,[(self.x,np.random.randn()),(self.y,np.random.randn())],1)
        self.output_node = node(-1,[(self.connector_node,np.random.randn())],2)
        self.inputLayer = layer([self.x, self.y])
        self.layer0 = layer([])
        self.layer1 = layer([self.connector_node])
        self.layer2 = layer([])
        self.outputLayer = layer([self.output_node])
        self.layerList = [self.inputLayer,self.layer0,self.layer1,self.layer2,self.outputLayer]
    def forward_pass(self, x, y):
        calculated_nodes = {0:x,1:y}
        for layer in self.layerList:
            if layer != self.inputLayer and layer != self.outputLayer and layer.nodeList != []:
                for node in layer:
                    id = layer[node].id
                    value = 0
                    for input in node.con_in:
                        in_id = input[0]
                        weight = input[1]
                        value += calculated_nodes.get(in_id)*weight
                    calculated_nodes.update({id:value})
        return  self.output_node.value

    # Choose a random layer and a random node in it
    # Take a random in node from it and replace it
    # Put this node in the layer between the in and out node set in node to original weight, out node to 1
    # If there is no layer in between the nodes, make a new layer for it
    def mutate_add_node(self):
        chosenLayerIndex = np.random.randint(1,len(self.layerList)-1)
        chosenLayer = self.layerList[chosenLayerIndex]
        outNode = chosenLayer[np.random.randint(0,len(self.layerList[chosenLayer])-1)]
        inNodeIndex = np.random.randint(0,len(outNode.con_in)-1)
        connectionTuple = outNode.con_in[inNodeIndex]
        inNode = connectionTuple[0]
        weight = connectionTuple[1]
        newNode = node(self.global_node_id,[(inNode,weight)],chosenLayerIndex)
        # Add new node to the layer below and set its in node list to the weight and input node of the old connection
        self.layerList[chosenLayer-1].append(newNode)
        self.global_node_id += 1
        # update the old connection to the new one for the out node with weight 1
        outNode.con_in[inNodeIndex] = (newNode,1)

    # Choose a random layer and a random node in it
    # Check if that isnt fully connected to the layer below
    # Add a connection from that node to an other and add a random weight to it
    def mutate_add_connection(self):
        chosenLayerIndex = np.random.randint(1, len(self.layerList) - 1)
        chosenLayer = self.layerList[chosenLayerIndex]
class layer:
    def __init__(self, nodes):
        self.nodedict = {}
        for i in nodes:
            self.nodedict.update({i.id:i})
class node:
    def __init__(self, id, in_list,layer_id):
        # id of the current node
        self.id= id
        # tuple (in node, weight)
        self.con_in = in_list
        # layer number
        self.layer_id = layer_id
        # value of the node
        self.value = 0