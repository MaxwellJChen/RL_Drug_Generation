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

class reinforce_policy(nn.Module):
    """
    Policy network for reinforce training
    """

    def __init__(self, num_node_features, global_vector_dim = 32):
        super(reinforce_policy, self).__init__()
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

        # Make weight initialization orthogonal

        # Initialize policy output layer weights with a scale of 0.01

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
        t = self.t_fcn1(g) # Logits
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

    def act(self, batch):
        """
        Accepts a batch of graphs. Outputs a list of actions for each state in batch alongside with log probability.
        """

        t, n, b = self.forward(batch.x, batch.edge_index, batch.batch) # Generate probability distributions for each action

        # Termination
        t = Categorical(t)
        t_act = t.sample()
        t_log_prob = t.log_prob(t_act)

        n1_act = 0
        n1_log_prob = 0
        n2_act = 0
        n2_log_prob = 0
        # Select two atoms: one from the original molecule and another from the every possible atom.
        state_idx, num_nodes = torch.unique(batch.batch, return_counts = True)
        for i in range(len(num_nodes)):
            # Select first atom from existing molecule
            full_idx = sum(num_nodes[:i+1]) # The length of the entire molecule and the atom bank
            prev_idx = sum(num_nodes[:i])

            prob_molecule = n[prev_idx:full_idx-10].view(-1)
            prob_molecule = prob_molecule.softmax(dim = 0)
            c_molecule = Categorical(prob_molecule)
            i_molecule = c_molecule.sample()
            i_molecule_log_prob = c_molecule.log_prob(i_molecule)

            # Create new categorical distribution without i_molecule that includes atom bank
            prob_full = n[prev_idx:full_idx].view(-1)
            prob_full = torch.cat((prob_full[:i_molecule], prob_full[i_molecule + 1:])) # Removing i_molecule with indexing
            prob_full = prob_full.softmax(dim = 0)
            c_full = Categorical(prob_full)
            i_full = c_full.sample()
            i_full_log_prob = c_full.log_prob(i_full)

            if i_full >= i_molecule:
                i_full += 1
                n1_act = i_molecule
                n2_act = i_full
                n1_log_prob = i_molecule_log_prob
                n2_log_prob = i_full_log_prob
            else:
                n2_act = i_molecule
                n1_act = i_full
                n2_log_prob = i_molecule_log_prob
                n1_log_prob = i_full_log_prob

        # Bond
        b = Categorical(b)
        b_act = b.sample()
        b_log_prob = b.log_prob(b_act)

        return int(t_act[0]), t_log_prob, int(n1_act), n1_log_prob, int(n2_act), n2_log_prob, int(b_act[0]), b_log_prob