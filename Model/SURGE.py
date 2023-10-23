import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool, BatchNorm
import rdkit.Chem as Chem

import numpy as np

torch.manual_seed(42)

def _fcn_init(fcn_layer, std = np.sqrt(2), bias_const = 0.0):
    """
    Helper function for initializing layers with orthogonal weights and constant bias.
    """
    torch.nn.init.orthogonal_(fcn_layer.weight, std)
    torch.nn.init.constant_(fcn_layer.bias, bias_const)
    return fcn_layer

def graph_softmax(logits, num_nodes):
    """
    Returns softmax of output probability distribution with multiple graphs.
    """
    logits = torch.split(logits, num_nodes.tolist())
    logits = [F.softmax(l, dim=0) for l in logits]
    logits = torch.cat(logits, dim=0)
    return logits

def graph_mean(tensor, num_nodes):
    tensor = torch.split(tensor, num_nodes.tolist())
    tensor_mean = [torch.mean(t) for t in tensor]
    tensor_mean = torch.tensor(tensor_mean).view(len(tensor_mean), 1)
    return tensor_mean

def graph_std(tensor, num_nodes):
    tensor = torch.split(tensor, num_nodes.tolist())
    tensor_std = []
    for t in tensor:
        if len(t) == 1:
            tensor_std.append(t * 0)
        else:
            tensor_std.append(torch.std(torch.flatten(t)))
    tensor_std = torch.tensor(tensor_std)
    tensor_std = tensor_std.view(len(tensor_std), 1)
    return tensor_std

class SURGE(nn.Module):
    """
    Actor-critic GNN model for computing actions based on molecular graph input.
    """

    def __init__(self, num_node_features = 24, policy_hidden_dim = 32, value_hidden_dim = 32):
        super(SURGE, self).__init__()

        # Policy embedder
        self.p_conv1 = GATv2Conv(num_node_features, 64, heads=3)
        self.p_bnorm1 = BatchNorm(64*3)
        self.p_conv2 = GATv2Conv(64*3, 64, head=3)
        self.p_bnorm2 = BatchNorm(64)
        self.p_conv3 = GATv2Conv(64, policy_hidden_dim, head=3)
        self.p_bnorm3 = BatchNorm(policy_hidden_dim)

        # Nmol
        self.nmol_fcn1 = nn.Linear(policy_hidden_dim, 32)
        self.nmol_fcn2 = nn.Linear(32, 16)
        self.nmol_fcn3 = nn.Linear(16, 1)

        # Nfull
        self.nfull_fcn1 = nn.Linear(policy_hidden_dim + 1, 16)
        self.nfull_fcn2 = nn.Linear(16, 1)

        # Bond
        self.b_fcn1 = nn.Linear(policy_hidden_dim*3, 16)
        self.b_fcn2 = nn.Linear(16, 3)

        # Termination
        self.t_fcn1 = nn.Linear(policy_hidden_dim*3 + 3 + 2 + 2 + 2 + 1, 2)

        # Value
        self.v_conv1 = GATv2Conv(num_node_features, 64)
        self.v_conv2 = GATv2Conv(64, 64)
        self.v_conv3 = GATv2Conv(64, value_hidden_dim)

        # Value predictor
        self.v_fcn1 = nn.Linear(value_hidden_dim, 16)
        self.v_fcn2 = nn.Linear(16, 1)

        # Masking variables
        self.max_valences = {'C': 4, 'O': 2, 'N': 3, 'S': 6, 'F': 1, 'Cl': 1, 'P': 5, 'Br': 1, 'I': 1, 'B': 3}
        self.atom_bank = ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B']
        self.bond_bank = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]

    # Policy methods
    def policy_embed(self, batch):
        """
        Embeds a given graph for policy training.
        """
        p_x = self.p_conv1(batch.x, batch.edge_index)
        p_x = self.p_bnorm1(p_x)
        p_x = nn.LeakyReLU(0.2)(p_x)
        p_x = self.p_conv2(p_x, batch.edge_index)
        p_x = self.p_bnorm2(p_x)
        p_x = nn.LeakyReLU(0.2)(p_x)
        p_x = self.p_conv3(p_x, batch.edge_index)
        p_x = self.p_bnorm3(p_x)
        p_x = nn.LeakyReLU(0.2)(p_x)
        return p_x

    def nmol(self, states, batch, p_x):
        """
        Calculates the unnormalized probability that each of the atoms already in the molecule should form a bond.
        """
        state_idxs, num_nodes = torch.unique(batch.batch, return_counts=True)
        nmol = torch.split(p_x, num_nodes.tolist())
        nmol = [n[:-10] for n in nmol]
        nmol = torch.cat(nmol, dim=0)
        nmol = self.nmol_fcn1(nmol)
        nmol = nn.LeakyReLU(0.2)(nmol)
        nmol = self.nmol_fcn2(nmol)
        nmol = nn.LeakyReLU(0.2)(nmol)
        nmol = self.nmol_fcn3(nmol)
        nmol = nn.LeakyReLU(0.2)(nmol)
        nmol = graph_softmax(nmol, num_nodes - 10)

        return nmol

    def nfull(self, states, batch, p_x, nmol):
        """
        Calculates the unnormalized probability that a pre-existing atom or an atom from the atom bank should form
        a bond based on the probabilities from Nmol.
        """
        atom_bank_nmol = torch.full((10, 1), -1, dtype=torch.float32) # -1 signifying atom belongs to atom bank
        state_idxs, num_nodes = torch.unique(batch.batch, return_counts=True)
        nmol = torch.split(nmol, (num_nodes - 10).tolist())
        nmol = [torch.cat((n, atom_bank_nmol), dim=0) for n in nmol] # Add atom_bank_nmol in multiple places to nmol
        nmol = torch.cat(nmol, dim=0)

        p_x = torch.hstack((p_x, nmol)) # Concatenate vector of probabilities representing nmol predictions

        nfull = self.nfull_fcn1(p_x)
        nfull = nn.LeakyReLU(0.2)(nfull)
        nfull = self.nfull_fcn2(nfull)
        nfull = nn.LeakyReLU(0.2)(nfull)
        nfull = graph_softmax(nfull, num_nodes)

        return nfull

    def bond(self, states, batch, p_x, nmol, nfull):
        """
        Given the node embeddings and the predicted probabilities from Nmol and Nfull, creates an unnormalized
        probability mass function of bonds.
        """
        # Perform a weighted sum of nodes based on probabilities from Nmol
        state_idxs, num_nodes = torch.unique(batch.batch, return_counts=True)
        p_x_nmol = torch.split(p_x, num_nodes.tolist())
        p_x_nmol = [n[:-10] for n in p_x_nmol]
        p_x_nmol = torch.cat(p_x_nmol, dim=0)
        nmol = F.softmax(nmol, dim=0)
        nmol = torch.hstack([nmol for i in range(p_x_nmol.shape[1])])
        p_x_nmol = p_x_nmol * nmol
        batch_nmol = torch.split(batch.batch, num_nodes.tolist())
        batch_nmol = [b[:-10] for b in batch_nmol]
        batch_nmol = torch.cat(batch_nmol, dim=0)
        p_x_nmol = global_add_pool(p_x_nmol, batch_nmol)

        # Perform a weighted sum of nodes based on probabilities from Nfull
        nfull = torch.hstack([nfull for i in range(p_x.shape[1])])
        p_x_nfull = p_x * nfull
        p_x_nfull = global_add_pool(p_x_nfull, batch.batch)

        p = global_mean_pool(p_x, batch.batch)

        # Concatenate to globally embedded vector of original graph
        p_bond = torch.hstack((p, p_x_nmol, p_x_nfull))

        b = self.b_fcn1(p_bond)
        b = nn.LeakyReLU(0.2)(b)
        b = self.b_fcn2(b)
        b = nn.LeakyReLU(0.2)(b)
        b = F.softmax(b, dim=1)
        return b, p_bond

    def termination(self, states, batch, p_bond, nmol, nfull, bond):
        """
        Given the confidence of the Bond, Nmol, and Nfull predictions and a graph embedding, decides whether or not
        to termination generation based on mean and std of probability distributions.
        """
        state_idxs, num_nodes = torch.unique(batch.batch, return_counts=True)
        nmol_mean = graph_mean(nmol, num_nodes - 10)
        nmol_std = graph_std(nmol, num_nodes - 10)
        nfull_mean = graph_mean(nfull, num_nodes)
        nfull_std = graph_std(nfull, num_nodes)
        b_mean = torch.mean(bond, dim=1).view(len(num_nodes), 1)
        b_std = torch.std(bond, dim=1).view(len(num_nodes), 1)

        num_nodes = (num_nodes - 10).view(len(num_nodes), 1)

        p_t = torch.cat((p_bond, bond, nmol_mean, nmol_std, nfull_mean, nfull_std, b_mean, b_std, num_nodes), dim=1)

        t = self.t_fcn1(p_t)
        t = nn.LeakyReLU(0.2)(t)
        t = F.softmax(t, dim = 1)

        return t

    def policy(self, states, batch):
        p_x = self.policy_embed(batch)
        nmol = self.nmol(states, batch, p_x)
        nfull = self.nfull(states, batch, p_x, nmol)
        b, p_bond = self.bond(states, batch, p_x, nmol, nfull)
        t = self.termination(states, batch, p_bond, nmol, nfull, b)

        return t, nmol, nfull, b

    # Value methods
    def value(self, batch):
        """
        Computes the predicted value of a molecule for calculating GAEs during PPO training.
        """

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

    def forward(self, states, batch):
        t, nmol, nfull, b = self.policy(states, batch)
        v = self.value(batch)

        return t, nmol, nfull, b, v

    def act(self, states, batch):
        """
        Returns a dictionary of actions for Termination, Nmol, Nfull, and Bond. These actions can be processed
        by the MolEnv to update a molecule. The actions are based on the probabilities returned by the forward
        method.
        Since the actions are sampled, not deterministically chosen based on the action with the greatest
        probability, there is some stochasticity in the resulting selections.
        When training, the method also returns the log probabilities of the selected actions.
        """

        # Obtain probabilities for the 4 separate probability distributions
        t, nmol, nfull, b = self.policy(states, batch)

        # Termination sampling
        t_categorical = Categorical(t)
        t_actions = t_categorical.sample()
        if self.training: # Record the log probabilities if specified
            t_log_probs = t_categorical.log_prob(t_actions)
        t_actions = t_actions.tolist()

        # Nmol and nfull sampling
        nmol_actions = [] # Initialize lists to hold actions for both nmol and nfull
        nfull_actions = []

        if self.training:
            nmol_log_probs = []
            nfull_log_probs = []

        state_idxs, num_nodes = torch.unique(batch.batch, return_counts = True) # Must split the nodes of graphs into separate probability distributions
        num_full = num_nodes.tolist()
        num_mol = [num - 10 for num in num_nodes.tolist()] # Nmol excludes atoms from the atom bank
        nmol_logits = torch.split(nmol, num_mol)
        nfull_logits = torch.split(nfull, num_full)

        for i, (nmol_single, nfull_single) in enumerate(zip(nmol_logits, nfull_logits)):
            nmol_categorical = Categorical(nmol_single.squeeze(dim = 1))
            nmol_action = nmol_categorical.sample()
            nmol_actions += [nmol_action.item()]

            nfull_categorical = Categorical(nfull_single.squeeze(dim = 1))
            nfull_action = nfull_categorical.sample()
            nfull_actions += [nfull_action.item()]

            if self.training: # Must calculate log probabilities in the for loop
                nmol_log_probs += [nmol_categorical.log_prob(nmol_action)]
                nfull_log_probs += [nfull_categorical.log_prob(nfull_action)]

        if self.training:
            nmol_log_probs = torch.hstack(nmol_log_probs)
            nfull_log_probs = torch.hstack(nfull_log_probs)

        # Bond sampling
        b_categorical = Categorical(b)
        b_actions = b_categorical.sample()
        if self.training:
            b_log_probs = b_categorical.log_prob(b_actions)
        b_actions = b_actions.tolist()

        # Store all the actions in a dictionary
        actions = {"t": t_actions, "nmol": nmol_actions, "nfull": nfull_actions, "b": b_actions}

        if self.training: # Store all the log probabilities in a dictionary
            log_probs = {"t": t_log_probs, "nmol": nmol_log_probs, "nfull": nfull_log_probs, "b": b_log_probs}
            return actions, log_probs

        return actions

if __name__ == '__main__':
    import rdkit.Chem as Chem
    from rdkit.Chem import RWMol
    import torch
    import torch
    import torch.nn as nn
    from Model.graph_embedding import batch_from_smiles, batch_from_states
    from Reinforcement_Learning.mol_env import vectorized_mol_env

    model = SURGE()
    smiles = ['c1ccccc1', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'] # 6 atoms in benzene, 24 in caffeine
    batch = batch_from_smiles(smiles)
    model.eval()
    env = vectorized_mol_env(max_steps = 200)
    states = env.reset()
    for i in range(1, 301):
        actions = model.act(batch_from_states(states))
        states, rewards, valids, timestep = env.step(actions['t'], actions['nmol'], actions['nfull'], actions['b'])
        print(f'{i}:\t{rewards}\t{states}')
        if i % 200 == 0:
            states = env.reset()