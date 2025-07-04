# This is a simple python (.py) file and not a .ipynb file because this file will later be imported for training.

'''
The file is a boilerplate for us to efficiently create graph convolutional networks without creating everything from scratch.
Although, some of the modules are available as methods in the torch library, we use our code for better understanding and customisation purposes as this project requires us
to have inputs of varying degrees.
The following libraries need to installed for this file: torch, torch.nn and torch_scatter
Abbreviations: GNN: Graph Neural Network
	       GCN: Graph Convolution Network
	       DTNN: Deep Tensor Neural Network
	       DAG: Directed Acyclic Graph

Authors: Apurva Goel (apurvago)
Documentation/ Comments: Amitha Kethavath (amithake)

'''


# Step 1: Importing Libraries
import torch
import torch.nn as nn
from torch_scatter import scatter_mean                   



#Step 2: Defining Graphs
'''
Sequential Graphs:
This is the foundational class for our GNNs.
It contains the schema of the Graphs such as Nodes, edges, and other metadata.
'''
class SequentialGraph(nn.Module):
    def __init__(self, input_dim):                                                             # Constructor Definition
        super(SequentialGraph, self).__init__()
        self.layers = nn.ModuleList()

    def add(self, layer):								      # Method to Add layers in the Neural Network
        self.layers.append(layer)

    def forward(self, data):								       # Forward Pass
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            if hasattr(layer, 'requires_graph') and layer.requires_graph:
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x



'''
SequentialDTNNGraph:
This type of graph is used when we need to use the distance between two connnected atoms. The distance mentioned in the adjacency list would be the absolute distance and not the one 
required for this type of graph.
'''
class SequentialDTNNGraph(nn.Module):
    def __init__(self):										# Constructor Definition
        super(SequentialDTNNGraph, self).__init__()
        self.layers = nn.ModuleList()

    def add(self, layer):									# Adding layers to the graph/ network.
        self.layers.append(layer)

    def forward(self, atom_numbers, distances, membership):					# Forward Pass.
        x = atom_numbers
        for layer in self.layers:
            if hasattr(layer, 'requires_graph') and layer.requires_graph:
                if isinstance(layer, DTNNStep):
                    x = layer(x, distances)
                elif isinstance(layer, DTNNGather):
                    x = layer(x, membership)
            else:
                x = layer(x)
        return x



'''
SequentialDAGGraph:
This graph treats every molecule is a Directed Acyclic Graph rooted at each atomic level.
'''
class SequentialDAGGraph(nn.Module):
    def __init__(self):										# Constructor Defintion
        super(SequentialDAGGraph, self).__init__()
        self.layers = nn.ModuleList()

    def add(self, layer):									# Addition of layers in the network.
        self.layers.append(layer)

    def forward(self, x, dag_topology, membership):						# Forward Pass
        for layer in self.layers:
            if isinstance(layer, DAGLayer):
                x = layer(x, dag_topology)
            elif isinstance(layer, DAGGather):
                x = layer(x, membership)
            else:
                x = layer(x)
        return x
        
        
'''
SequentialWeaveGraph:
This is used when the chemical we are observing needs to be analysed from a pairwise perspective i.e., the molecules are chemically bonded.
'''
class SequentialWeaveGraph(nn.Module):							
    def __init__(self):									# Constructor Definition
        super(SequentialWeaveGraph, self).__init__()
        self.layers = nn.ModuleList()

    def add(self, layer):								# Adding layers to the graph.
        self.layers.append(layer)

    def forward(self, x, pair_features, pair_index, atom_mask=None, membership=None):	# Forward Pass
        for layer in self.layers:
            if isinstance(layer, WeaveLayer):
                x, pair_features = layer(x, pair_features, pair_index)
            elif isinstance(layer, WeaveConcat):
                x = layer(x, atom_mask)
            elif isinstance(layer, WeaveGather):
                x = layer(x, membership)
            else:
                x = layer(x)
        return x



'''
AlternateSequentialWeaveGraph:
This is an alternative we use for testing purposes.
'''
class AlternateWeaveLayer(nn.Module):
    requires_graph = True

    def __init__(self, atom_in_dim, pair_in_dim, atom_out_dim, pair_out_dim):
        super(AlternateWeaveLayer, self).__init__()
        self.atom_transform = nn.Sequential(
            nn.Linear(atom_in_dim, atom_out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(atom_out_dim)
        )
        self.pair_transform = nn.Sequential(
            nn.Linear(pair_in_dim, pair_out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(pair_out_dim)
        )
        self.atom_to_pair = nn.Linear(atom_in_dim * 2, pair_out_dim)

    def forward(self, x, pair_features, pair_index):
        atom_out = self.atom_transform(x)
        send = x[pair_index[0]]
        recv = x[pair_index[1]]
        pair_input = torch.cat([send, recv], dim=-1)
        pair_update = self.atom_to_pair(pair_input) + self.pair_transform(pair_features)
        return atom_out, pair_update

class AlternateWeaveGather(nn.Module):
    requires_graph = True

    def __init__(self, input_dim, output_dim):
        super(AlternateWeaveGather, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, batch):
        x = self.linear(x)
        return scatter_mean(x, batch, dim=0)


class AlternateSequentialWeaveGraph(nn.Module):
    def __init__(self):
        super(AlternateSequentialWeaveGraph, self).__init__()
        self.layers = nn.ModuleList()

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x, pair_features, pair_index, batch):
        for layer in self.layers:
            if isinstance(layer, AlternateWeaveLayer):
                x, pair_features = layer(x, pair_features, pair_index)
            elif isinstance(layer, AlternateWeaveGather):
                x = layer(x, batch)
            else:
                x = layer(x)
        return x
        
'''
SequentialSupportGraph:
This is used for one-shot learning. This is highly effective as after witnessing just one example, the model begins to classify or predict the nature of other examples.
'''
class SequentialSupportGraph(nn.Module):					
    def __init__(self, input_dim):						# Constructor Defintion
        super(SequentialSupportGraph, self).__init__()
        self.test_layers = nn.ModuleList()
        self.support_layers = nn.ModuleList()
        self.test = None
        self.support = None

    def add(self, layer):							# Adding layers to the graph.
        self.test_layers.append(layer)
        self.support_layers.append(layer)

    def add_test(self, layer):							# Adding layers for test data. Optional
        self.test_layers.append(layer)
        
        
    def add_support(self, layer):						# Adding layers for support.
        self.support_layers.append(layer)

    def join(self, layer):							#Joining of layers.
        self.join_layer = layer

    def forward(self, test_data, support_data):					# Forward Pass
        x_test, x_support = test_data, support_data
        for test_layer, support_layer in zip(self.test_layers, self.support_layers):
            if hasattr(test_layer, 'requires_graph') and test_layer.requires_graph:
                x_test = test_layer(*x_test)
                x_support = support_layer(*x_support)
            else:
                x_test = test_layer(x_test)
                x_support = support_layer(x_support)

        if hasattr(self, 'join_layer'):
            x_test, x_support = self.join_layer(x_test, x_support)

        return x_test, x_support


