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

    def act(self, batch):
        """
        Batch of graphs from states. List of states. States are RWMol or Mol.
        """

        t_logits, nmol_logits, nfull_logits, b_logits = self.policy(batch)



    # def act(self, batch, states):
    #     """
    #     Batch of graphs from states. List of states. States are RWMol or Mol.
    #     """
    #
    #     t_logits, nmol_logits, nfull_logits, b_logits = self.policy(batch)
    #
    #     # Empty masks for termination and bond. Nmol and Nfull are easier made after torch.split and iterating through individual graphs.
    #     t_mask = torch.ones(t_logits.size())
    #
    #     nmol_logits = torch.squeeze(nmol_logits)
    #     nfull_logits = torch.squeeze(nfull_logits)
    #
    #     nmol_sizes = [state.GetNumHeavyAtoms() for state in states] # List of the number of nodes in each graph in nmol
    #     nfull_sizes = [size + 10 for size in nmol_sizes] # List of number of nodes in each graph for nfull
    #
    #     nmol_masks = []
    #     nfull_masks = []
    #
    #     nmol_act = []
    #     nmol_log_prob = []
    #     nfull_act = []
    #     nfull_log_prob = []
    #     for i, (nmol_single, nfull_single) in enumerate(zip(torch.split(nmol_logits, nmol_sizes), torch.split(nfull_logits, nfull_sizes))):
    #         # Create empty masks
    #         nmol_mask = torch.ones(nmol_single.size())
    #         nfull_mask = torch.ones(nfull_single.size())
    #
    #         has_max_valence = [self.max_valences[atom.GetSymbol()] == sum([int(bond.GetBondType()) for bond in atom.GetBonds()]) - atom.GetFormalCharge() for atom in states[i].GetAtoms()] # List of booleans representing whether an atom in the original molecule has been bonded to its maximal valence
    #         valence_diffs = [self.max_valences[atom.GetSymbol()] - sum([int(bond.GetBondType()) for bond in atom.GetBonds()]) - atom.GetFormalCharge() for atom in states[i].GetAtoms()] # List of ints representing difference in maximum valence of an atom and the actual valence
    #
    #         if all(has_max_valence): # 1. All atoms are at their max valence. Must terminate generation.
    #             t_mask[i][0] = 0
    #         for j in range(len(has_max_valence)): # 2/3. A specific atom is at its max valence, so it cannot be chosen.
    #             if has_max_valence[j]:
    #                 nmol_mask[j] = 0
    #                 nfull_mask[j] = 0
    #
    #         nmol_masks.append(nmol_mask)
    #         nmol_mask_bool = nmol_mask.type(torch.BoolTensor)
    #         nmol_masked = torch.where(nmol_mask_bool, nmol_single, torch.tensor(-1e10))
    #         nmol_categorical = Categorical(logits = nmol_masked)
    #         nmol_single_act = nmol_categorical.sample()
    #         nmol_log_prob += [nmol_categorical.log_prob(nmol_single_act).item()]
    #         nmol_single_act = nmol_single_act.item()
    #         nmol_act += [nmol_single_act]
    #
    #         # 4. Cannot select the same atom in nfull as in nmol
    #         nfull_mask[nmol_single_act] = 0
    #
    #         ring_info = states[i].GetRingInfo()
    #         for j in range(states[i].GetNumHeavyAtoms()): # Iterate through all the atoms in the molecule
    #             if j != nmol_single_act and not has_max_valence[j]: # If atom j is not the same as nmol_single_act and can form a bond
    #
    #                 # 5. Both atoms are not in the same ring
    #                 if len(ring_info.AtomRings()) != 0 and ring_info.AreAtomsInSameRing(j, nmol_single_act):  # Cannot make a bond between atoms in the same ring.
    #                     nfull_mask[j] = 0
    #
    #                 # 6. Both atoms are not already bonded
    #                 elif states[i].GetBondBetweenAtoms(j, nmol_single_act) is not None: # Cannot make a bond between atoms if they already have a bond.
    #                     nfull_mask[j] = 0
    #
    #                 # 7. Forming a bond between the atoms would not result in a ring size smaller than 3 or greater than 7
    #                 elif not has_max_valence[j]: # Atom j is capable of forming a bond.
    #                     test_state = RWMol(copy.copy(states[i]))
    #                     test_state.AddBond(nmol_single_act, j, order = self.bond_bank[0]) # Form a hypothetical single bond between the atoms
    #                     Chem.SanitizeMol(test_state, catchErrors=True)
    #                     test_state.UpdatePropertyCache()
    #                     test_ring_info = test_state.GetRingInfo() # Assuming a bond of the smallest order between j and nmol_single_act is formed and has rings
    #                     if len(test_ring_info.AtomRings()) != 0:
    #                         test_ring_sizes = [len(ring) for ring in test_ring_info.AtomRings()]
    #                         if max(test_ring_sizes) > 7: # Ensuring that if a ring is formed by the connection, the ring contains 3 to 7 atoms
    #                             nfull_mask[j] = 0
    #
    #         nfull_masks.append(nfull_mask)
    #         nfull_mask_bool = nfull_mask.type(torch.BoolTensor)
    #         nfull_masked = torch.where(nfull_mask_bool, nfull_single, torch.tensor(-1e10))
    #         nfull_categorical = Categorical(logits = nfull_masked)
    #         nfull_single_act = nfull_categorical.sample()
    #         nfull_log_prob += [nfull_categorical.log_prob(nfull_single_act).item()]
    #         nfull_single_act = nfull_single_act.item()
    #         nfull_act += [nfull_single_act]
    #
    #     t_mask_bool = t_mask.type(torch.BoolTensor)
    #     t_masked = torch.where(t_mask_bool, t_logits, torch.tensor(-1e10))
    #     t_categorical = Categorical(logits=t_masked)
    #     t_act = t_categorical.sample() # Categorical outputs a tensor. Convert to list instead.
    #     t_log_prob = t_categorical.log_prob(t_act).tolist()
    #     t_act = t_act.tolist()
    #
    #     b_mask = torch.ones(b_logits.size())
    #
    #     # 8. Bond cannot be greater than the least remaining valence among the selected atoms
    #     for i in range(len(states)):
    #         nmol_valence = self.max_valences[states[i].GetAtomWithIdx(nmol_act[i]).GetSymbol()] - sum([int(bond.GetBondType()) for bond in states[i].GetAtomWithIdx(nmol_act[i]).GetBonds()]) # Calculating remaining valences of nmol and nfull
    #         if nfull_act[i] >= states[i].GetNumHeavyAtoms(): # Nfull adds an atom to the molecule
    #             nfull_valence = self.max_valences[self.atom_bank[nfull_act[i] - states[i].GetNumHeavyAtoms()]] # Valence of additional atom is the same as the max valence of that element
    #         else: # Nfull is in the original molecule
    #             nfull_valence = self.max_valences[states[i].GetAtomWithIdx(nfull_act[i]).GetSymbol()] - sum([int(bond.GetBondType()) for bond in states[i].GetAtomWithIdx(nmol_act[i]).GetBonds()])
    #         for j in range(min(nmol_valence, nfull_valence), 3):
    #             b_mask[i][j] = 0
    #
    #     b_mask_bool = b_mask.type(torch.BoolTensor)
    #     b_masked = torch.where(b_mask_bool, b_logits, torch.tensor(-1e10))
    #     b_categorical = Categorical(logits = b_masked)
    #     b_act = b_categorical.sample()
    #     b_log_prob = b_categorical.log_prob(b_act).tolist()
    #     b_act = b_act.tolist()
    #
    #     return t_act, t_log_prob, t_mask, nmol_act, nmol_log_prob, nmol_masks, nfull_act, nfull_log_prob, nfull_masks, b_act, b_log_prob, b_mask

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

    surge = SURGE
    smiles = ['c1ccccc1', 'CC']
    batch = batch_from_smiles(smiles)
    print(surge.policy(batch))