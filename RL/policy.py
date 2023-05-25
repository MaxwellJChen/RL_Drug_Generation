import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions import Categorical

import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool

import graph_embedding

import rdkit.Chem as Chem

import numpy as np
import random

class Policy(nn.Module):
    def __init__(self, num_node_features, global_vector_dim = 32):
        super(Policy, self).__init__()
        # Global embedder
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 24)
        self.g_fcn1 = nn.Linear(24, global_vector_dim) # After global_mean_pool()

        # Termination predictor
        self.t_fcn1 = nn.Linear(global_vector_dim, 2)

        # Node predictor
        self.n_fcn1 = nn.Linear(24, 16)
        self.n_fcn2 = nn.Linear(16, 1)

        # Bond predictor
        self.b_fcn1 = nn.Linear(24, 16)
        self.b_fcn2 = nn.Linear(16, 3)

    def forward(self, x, edge_index, batch):
        """
        Forward function of policy model. Operates on a batch of graphs.
        :return: t – probability distribution of termination, n – probability of node selections, b – probability of bond
        """

        # Graph embedding
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index) # Node embeddings
        x = x.relu()

        g = global_mean_pool(x, batch) # Globally embedded vector
        g = self.g_fcn1(g)

        # Termination prediction
        t = self.t_fcn1(g)
        t = t.softmax(dim = 1) # Probability distribution of termination for each molecule in batch

        # Atom prediction
        n = self.n_fcn1(x)
        n = n.relu()
        n = self.n_fcn2(n)
        n = torch.vstack([n[torch.where(batch == g)].softmax(dim = 0) for g in torch.unique(batch)]) # Batch-wise softmax

        # Bond prediction
        b = n * x # Weighs each node by probability of bond formation
        b = global_add_pool(b, batch)
        b = self.b_fcn1(b)
        b = b.relu()
        b = self.b_fcn2(b)
        b = b.softmax(dim = 0) # Probability distribution of 3 bonds for each molecule in batch

        return t, n, b

    def act(self, state):
        """
        Accepts a batch of states. Outputs a list of actions for each state in batch.
        """

        # Sample nodes for each example in batch
        t, n, b = self.forward(state)
        t = Categorical(t)


    #     selected_idx = []
    #     for i in range(len(num_nodes)):
    #         # Select first atom from existing molecule
    #         full_idx = sum(num_nodes[:i+1]) # The length of the entire molecule and the atom bank
    #         prev_idx = sum(num_nodes[:i])
    #
    #         prob_molecule = n[prev_idx:full_idx-10].view(-1)
    #         prob_molecule = prob_molecule.softmax(dim = 0)
    #         c_molecule = Categorical(prob_molecule)
    #         i_molecule = int(c_molecule.sample())
    #
    #         # Create new categorical distribution without i_molecule that includes atom bank
    #         prob_full = n[prev_idx:full_idx].view(-1)
    #         prob_full = torch.cat((prob_full[:i_molecule], prob_full[i_molecule + 1:])) # Removing i_molecule with indexing
    #         prob_full = prob_full.softmax(dim = 0)
    #         c_full = Categorical(prob_full)
    #         i_full = int(c_full.sample())
    #
    #         # Updating indices to apply to stack
    #         i_molecule += prev_idx
    #         i_full += prev_idx
    #
    #         if i_full > i_molecule:
    #             selected_idx.append([i_molecule, i_full])
    #         else:
    #             selected_idx.append([i_full, i_molecule])
    #
    #     # Combine selected node embeddings
    #     selected_nodes = torch.empty([len(num_nodes), 128])
    #     for i in range(len(selected_idx)):
    #         selected_nodes[i] = torch.hstack((stack[selected_idx[i][0]], stack[selected_idx[i][1]]))


"""
Caffeine: Cn1cnc2n(C)c(=O)n(C)c(=O)c12
Ibuprofen: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
Benzene: c1ccccc1
"""

smiles = ["Cn1cnc2n(C)c(=O)n(C)c(=O)c12", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "c1ccccc1"]
data_list = graph_embedding.graph_from_smiles_atom_bank(smiles)
batch = torch_geometric.data.Batch.from_data_list(data_list)

policy = Policy(batch.num_node_features)
t, n, b = policy(batch.x, batch.edge_index, batch.batch)
print(n.shape)

# x_smiles =
# data_list, num_nodes = ge.graph_from_smiles_atom_bank_with_list(["Cn1cnc2n(C)c(=O)n(C)c(=O)c12"])
# data_list = torch_geometric.data.Batch.from_data_list(data_list)
#
# policy = Policy(data_list.num_node_features)
# t, n, b = policy(data_list.x, data_list.edge_index, num_nodes, data_list.batch)
# print(n)