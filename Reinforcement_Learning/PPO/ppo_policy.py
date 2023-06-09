import torch
import torch.nn as nn
from torch.distributions import Categorical

from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_add_pool

import numpy as np

torch.manual_seed(42)

def _fcn_init(fcn_layer, std = np.sqrt(2), bias_const = 0.0):
    """
    Helper function for initializing layers with orthogonal weights and constant bias.
    """

    torch.nn.init.orthogonal_(fcn_layer.weight, std)
    torch.nn.init.constant_(fcn_layer.bias, bias_const)
    return fcn_layer

def _graph_init(graph_layer, std = np.sqrt(2), bias_const = 0.0):
    """
    Helper function for initializing GNN layers with orthogonal weights and constant bias.
    """
    pass

class ppo_policy(nn.Module):
    """
    Policy network for PPO training
    """
    def __init__(self, num_node_features, global_vector_dim = 32):
        super(ppo_policy, self).__init__()
        # Global embedder
        self.conv1 = GatedGraphConv(num_node_features, 10)
        self.conv2 = GatedGraphConv(32, 10)
        self.conv3 = GatedGraphConv(32, 10)
        self.g_fcn1 = _fcn_init(nn.Linear(32, 24))
        self.g_fcn2 = _fcn_init(nn.Linear(24, global_vector_dim))

        # Termination predictor
        self.t_fcn1 = _fcn_init(nn.Linear(global_vector_dim, 32))
        self.t_fcn2 = _fcn_init(nn.Linear(32, 2))

        # Node predictor
        self.n_fcn1 = _fcn_init(nn.Linear(32, 16))
        self.n_fcn2 = _fcn_init(nn.Linear(16, 1))

        # Bond predictor
        self.b_fcn1 = _fcn_init(nn.Linear(32, 16))
        self.b_fcn2 = _fcn_init(nn.Linear(16, 3))

    def forward(self, x, edge_index, batch):
        """
        Forward function of policy model. Operates on a batch of graphs.
        :return: t – probability distribution of termination, n – probability of node selections, b – probability of bond
        """

        # Graph embedding
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)  # Node embeddings
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()

        g = global_mean_pool(x, batch)  # Globally embedded vector
        g = self.g_fcn1(g)
        g = self.g_fcn2(g)

        # Termination prediction
        t = self.t_fcn1(g)
        t = t.relu()
        t = self.t_fcn2(t)
        t = t.relu()

        # Atom prediction
        n = self.n_fcn1(x)
        n = n.relu()
        n = self.n_fcn2(n)
        n = n.relu()

        # Bond prediction
        b = global_add_pool(x, batch)
        b = self.b_fcn1(b)
        b = b.relu()
        b = self.b_fcn2(b)
        b = b.relu()

        return t, n, b # Returns logits

    def act(self, batch, mol_env, return_masks = False):
        """
        Accepts a batch of graphs. Outputs a list of actions for each state in batch alongside with log probability.
        """

        t, n, b = self.forward(batch.x, batch.edge_index,
                               batch.batch)  # Generate probability distributions for each action

        # Record the masks applied for training PPO
        t_masks = []
        nmol_masks = []
        nfull_masks = []
        b_masks = []

        # Termination
        t = t.softmax(dim=1)
        t = Categorical(t) # No invalid action masking needed for termination
        t_act = t.sample()
        t_log_prob = t.log_prob(t_act)

        # Record processed actions to send directly to environment
        n1_act = []
        n2_act = []

        # Record the outputs of the categorical distributions
        nmol_act = []
        nmol_log_prob = []
        nfull_act = []
        nfull_log_prob = []

        # Select two atoms: one from the original molecule and another from the every possible atom.
        state_idx, num_nodes = torch.unique(batch.batch, return_counts=True)
        for i in range(len(num_nodes)): # Iterates through each graph in batch
            # Select first atom from existing molecule
            full_idx = sum(num_nodes[:i + 1])  # The length of the entire molecule and the atom bank
            prev_idx = sum(num_nodes[:i])

            prob_molecule = n[prev_idx:full_idx - 10].view(-1)

            # Invalid action masking for prob_molecule
            prob_mol_mask = [k for k, v in mol_env.has_max_valence.items() if v] # If the valence is already filled, cannot select an atom from the original molecule
            for idx in prob_mol_mask:
                prob_molecule[idx] = float(-1e10)
            # print(f'prob_molecule: {prob_molecule}')
            # print(f'prob_mol_mask: {prob_mol_mask}')

            prob_molecule = prob_molecule.softmax(dim=0)
            c_molecule = Categorical(prob_molecule)
            i_molecule = c_molecule.sample()
            i_molecule_log_prob = c_molecule.log_prob(i_molecule)
            nmol_act.append(i_molecule)
            nmol_log_prob.append(i_molecule_log_prob)

            # Create new categorical distribution that includes atom bank
            prob_full = n[prev_idx:full_idx].view(-1)

            # Invalid action masking for prob_full
            prob_full_mask = []
            prob_full_mask += prob_mol_mask # Cannot have a full valence
            prob_full_mask += [i_molecule] # Cannot be the same index as the first selected value
            for idx in range(mol_env.mol_size): # Cannot already have a bond
                if mol_env.state.GetBondBetweenAtoms(idx, int(i_molecule)) is not None:
                    prob_full_mask += [idx]
            ring_info = mol_env.state.GetRingInfo() # Cannot be in a ring with the first selected atom
            if len(ring_info.AtomRings()) != 0:
                for idx in range(len(mol_env.state.GetAtoms())):
                    if ring_info.AreAtomsInSameRing(idx, int(i_molecule)):
                        prob_full_mask += [idx]
            prob_full_mask = list(set([int(idx) for idx in prob_full_mask]))
            for idx in prob_full_mask:
                prob_full[idx] = float(-1e10)
            # print(f'prob_full: {prob_full}')
            # print(f'prob_full_mask: {prob_full_mask}')

            prob_full = prob_full.softmax(dim=0)
            c_full = Categorical(prob_full)
            i_full = c_full.sample()
            i_full_log_prob = c_full.log_prob(i_full)
            nfull_act.append(i_full)
            nfull_log_prob.append(i_full_log_prob)

            if i_full >= i_molecule:
                n1_act.append(i_molecule)
                n2_act.append(i_full) # Update i_full to account for i_molecule node being removed
            else:
                n2_act.append(i_molecule)
                n1_act.append(i_full)

            if return_masks:
                nmol_masks.append(prob_mol_mask)
                nfull_masks.append(prob_full_mask)

        # Bond
        # Invalid action masking for bond
        atoms = [atom for atom in mol_env.state.GetAtoms()]
        n1_valence = mol_env.atom_valences[int(n1_act[0])]
        n1_max_valence = mol_env.max_valences[atoms[int(n1_act[0])].GetSymbol()]
        if int(n2_act[0]) >= mol_env.mol_size: # Adding a new atom
            n2_valence = 0
            n2_max_valence = mol_env.max_valences[mol_env.atom_bank[int(n2_act[0]) - mol_env.mol_size].GetSymbol()]
        else:
            n2_valence = mol_env.atom_valences[int(n2_act[0])]
            n2_max_valence = mol_env.max_valences[atoms[int(n2_act[0])].GetSymbol()]
        allowed_bonds = range(min(n1_max_valence - n1_valence, n2_max_valence - n2_valence))
        bond_mask = [0, 1, 2]
        bond_mask = [idx for idx in bond_mask if idx not in allowed_bonds]
        for idx in bond_mask:
            b[0][idx] = float(-1e10)
        if return_masks:
            b_masks.append(bond_mask)

        b = b.softmax(dim = 1)
        b = Categorical(b)
        b_act = b.sample()
        b_log_prob = b.log_prob(b_act)

        t_act = [int(t) for t in list(t_act)]
        n1_act = [int(a1) for a1 in n1_act]
        n2_act = [int(a2) for a2 in n2_act]
        b_act = [int(b) for b in list(b_act)]

        if return_masks:
            return t_act, t_log_prob, t_masks, n1_act, n2_act, nmol_act, nmol_log_prob, nmol_masks, nfull_act, nfull_log_prob, nfull_masks, b_act, b_log_prob, b_masks
        else:
            return t_act, t_log_prob, n1_act, n2_act, nmol_act, nmol_log_prob, nfull_act, nfull_log_prob, b_act, b_log_prob