import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch_geometric
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_add_pool

import rdkit.Chem as Chem
from rdkit.Chem import RWMol

import numpy as np

from rdkit.Chem import Draw
import matplotlib.pyplot as plt

import copy

torch.manual_seed(42)

def _fcn_init(fcn_layer, std = np.sqrt(2), bias_const = 0.0):
    """
    Helper function for initializing layers with orthogonal weights and constant bias.
    """
    torch.nn.init.orthogonal_(fcn_layer.weight, std)
    torch.nn.init.constant_(fcn_layer.bias, bias_const)
    return fcn_layer

class SURGE(nn.Module):
    """
    Policy network for PPO training
    """
    def __init__(self, num_node_features = 24):
        super(SURGE, self).__init__()

        # Policy embedder
        self.p_conv1 = GatedGraphConv(num_node_features, 32)
        self.p_conv2 = GatedGraphConv(32, 32)
        self.p_conv3 = GatedGraphConv(32, 16)

        # Termination predictor
        self.t_fcn1 = _fcn_init(nn.Linear(32, 16))
        self.t_fcn2 = _fcn_init(nn.Linear(16, 2), std = 0.01)

        # Nmol predictor
        self.nmol_fcn1 = _fcn_init(nn.Linear(32, 16))
        self.nmol_fcn2 = _fcn_init(nn.Linear(16, 1), std = 0.01)

        # Nfull predictor
        self.nfull_fcn1 = _fcn_init(nn.Linear(32, 16))
        self.nfull_fcn2 = _fcn_init(nn.Linear(16, 1), std = 0.01)

        # Bond predictor
        self.b_fcn1 = _fcn_init(nn.Linear(32, 16))
        self.b_fcn2 = _fcn_init(nn.Linear(16, 3), std = 0.01)

        # Value embedder
        self.v_conv1 = GatedGraphConv(num_node_features, 32)
        self.v_conv2 = GatedGraphConv(32, 32)
        self.v_conv3 = GatedGraphConv(32, 16)

        # Value predictor
        self.v_fcn1 = _fcn_init(nn.Linear(32, 16))
        self.v_fcn2 = _fcn_init(nn.Linear(16, 1), std = 1)

        # Masking variables
        self.max_valences = {'C': 4, 'O': 2, 'N': 3, 'S': 6, 'F': 1, 'Cl': 1, 'P': 5, 'Br': 1, 'I': 1, 'B': 3}
        self.atom_bank = ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B']
        self.bond_bank = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]

    def forward(self, batch):
        # Policy embedding
        p_x = self.p_conv1(batch.x, batch.edge_index)
        p_x = nn.LeakyReLU(0.2)(p_x)
        p_x = self.p_conv2(p_x, batch.edge_index)
        p_x = nn.LeakyReLU(0.2)(p_x)
        p_x = self.p_conv3(p_x, batch.edge_index)
        p_x = nn.LeakyReLU(0.2)(p_x)

        p = global_mean_pool(p_x, batch.batch)

        # Termination prediction
        t = self.t_fcn1(p)
        t = nn.LeakyReLU(0.2)(t)
        t = self.t_fcn2(t)
        t = nn.LeakyReLU(0.2)(t)

        # Nmol prediction
        state_idxs, num_nodes = torch.unique(batch.batch, return_counts=True)
        nmol = torch.split(p_x, num_nodes.tolist())
        nmol = [n[:-10] for n in nmol]
        nmol = torch.cat(nmol, dim=0)
        nmol = self.nmol_fcn1(nmol)
        nmol = nn.LeakyReLU(0.2)(nmol)
        nmol = self.nmol_fcn2(nmol)
        nmol = nn.LeakyReLU(0.2)(nmol)

        # Nfull prediction
        nfull = self.nfull_fcn1(p_x)
        nfull = nn.LeakyReLU(0.2)(nfull)
        nfull = self.nfull_fcn2(nfull)
        nfull = nn.LeakyReLU(0.2)(nfull)

        # Bond prediction
        b = self.b_fcn1(p)
        b = nn.LeakyReLU(0.2)(b)
        b = self.b_fcn2(b)
        b = nn.LeakyReLU(0.2)(b)

        # Value embedding
        v_x = self.v_conv1(batch.x, batch.edge_index)
        v_x = nn.LeakyReLU(0.2)(v_x)
        v_x = self.v_conv2(v_x, batch.edge_index)
        v_x = nn.LeakyReLU(0.2)(v_x)
        v_x = self.v_conv3(v_x, batch.edge_index)
        v_x = nn.LeakyReLU(0.2)(v_x)

        v = global_mean_pool(v_x, batch.batch)

        # Value prediction
        v = self.v_fcn1(v)
        v = nn.LeakyReLU(0.2)(v)
        v = self.v_fcn2(v)
        v = nn.LeakyReLU(0.2)(v)

        return t, nmol, nfull, b, v

    def policy(self, batch):
        # Policy embedding
        p_x = self.p_conv1(batch.x, batch.edge_index)
        p_x = nn.LeakyReLU(0.2)(p_x)
        p_x = self.p_conv2(p_x, batch.edge_index)
        p_x = nn.LeakyReLU(0.2)(p_x)
        p_x = self.p_conv3(p_x, batch.edge_index)
        p_x = nn.LeakyReLU(0.2)(p_x)

        p = global_mean_pool(p_x, batch.batch)

        # Termination prediction
        t = self.t_fcn1(p)
        t = nn.LeakyReLU(0.2)(t)
        t = self.t_fcn2(t)
        t = nn.LeakyReLU(0.2)(t)

        # Nmol prediction
        state_idxs, num_nodes = torch.unique(batch.batch, return_counts=True)
        nmol = torch.split(p_x, num_nodes.tolist())
        nmol = [n[:-10] for n in nmol]
        nmol = torch.cat(nmol, dim=0)
        nmol = self.nmol_fcn1(nmol)
        nmol = nn.LeakyReLU(0.2)(nmol)
        nmol = self.nmol_fcn2(nmol)
        nmol = nn.LeakyReLU(0.2)(nmol)

        # Nfull prediction
        nfull = self.nfull_fcn1(p_x)
        nfull = nn.LeakyReLU(0.2)(nfull)
        nfull = self.nfull_fcn2(nfull)
        nfull = nn.LeakyReLU(0.2)(nfull)

        # Bond prediction
        b = self.b_fcn1(p)
        b = nn.LeakyReLU(0.2)(b)
        b = self.b_fcn2(b)
        b = nn.LeakyReLU(0.2)(b)

        return t, nmol, nfull, b # Logits

    def value(self, batch):
        # Value embedding
        v_x = self.v_conv1(batch.x, batch.edge_index)
        v_x = nn.LeakyReLU(0.2)(v_x)
        v_x = self.v_conv2(v_x, batch.edge_index)
        v_x = nn.LeakyReLU(0.2)(v_x)
        v_x = self.v_conv3(v_x, batch.edge_index)
        v_x = nn.LeakyReLU(0.2)(v_x)

        v = global_mean_pool(v_x, batch.batch)

        # Value prediction
        v = self.v_fcn1(v)
        v = nn.LeakyReLU(0.2)(v)
        v = self.v_fcn2(v)
        v = nn.LeakyReLU(0.2)(v)

        return v

    def act(self, batch, return_log_probs = False):
        # Obtain the unnormalized log probabilities for the 4 separate probability distributions
        t_logits, nmol_logits, nfull_logits, b_logits = self.policy(batch)

        # Termination sampling
        t_categorical = Categorical(logits = t_logits)
        t_actions = t_categorical.sample()
        if return_log_probs: # Record the log probabilities if specified
            t_log_probs = t_categorical.log_prob(t_actions)
        t_actions = t_actions

        # Nmol and nfull sampling
        nmol_actions = [] # Initialize lists to hold actions for both nmol and nfull
        nfull_actions = []

        state_idxs, num_nodes = torch.unique(batch.batch, return_counts = True) # Must split the nodes of graphs into separate probability distributions
        num_full = num_nodes.tolist()
        num_mol = [num - 10 for num in num_nodes.tolist()] # Nmol excludes atoms from the atom bank
        nmol_logits = torch.split(nmol_logits, num_mol)
        nfull_logits = torch.split(nfull_logits, num_full)

        for i, (nmol_single, nfull_single) in enumerate(zip(nmol_logits, nfull_logits)):
            nmol_categorical = Categorical(logits = nmol_single.squeeze(dim = 1))
            nmol_action = nmol_categorical.sample()
            nmol_actions += [nmol_action.item()]

            nfull_categorical = Categorical(logits = nfull_single.squeeze(dim = 1))
            nfull_action = nfull_categorical.sample()
            nfull_actions += [nfull_action.item()]

            if return_log_probs: # Must calculate log probabilities in the for loop
                nmol_log_probs = []
                nfull_log_probs = []
                nmol_log_probs += [nmol_categorical.log_prob(nmol_action)]
                nfull_log_probs += [nfull_categorical.log_prob(nfull_action)]

        # Bond sampling
        b_categorical = Categorical(logits = b_logits)
        b_actions = b_categorical.sample()
        if return_log_probs:
            b_log_probs = b_categorical.log_prob(b_actions)
        b_actions = b_actions.tolist()

        # Store all the actions in a dictionary
        actions = {"t": t_actions, "nmol": nmol_actions, "nfull": nfull_actions, "b": b_actions}

        if return_log_probs: # Store all the log probabilities in a dictionary
            log_probs = {"t": t_log_probs, "nmol": nmol_log_probs, "nfull": nfull_log_probs, "b": b_log_probs}

        return actions, log_probs

if __name__ == '__main__':
    import rdkit
    import rdkit.Chem as Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import RWMol
    import warnings
    import torch
    import random
    import torch
    import torch.nn as nn
    import copy
    import matplotlib.pyplot as plt
    from graph_embedding import batch_from_smiles, batch_from_states, visualize
    import pandas as pd

    model = SURGE()
    smiles = ['c1ccccc1', 'CC', 'CC', 'CC', 'c1ccccc1']
    batch = batch_from_smiles(smiles)
    print(model.act(batch))