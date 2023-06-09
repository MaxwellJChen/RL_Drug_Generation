import torch
import torch.optim as optim
import torch.nn as nn

import pandas as pd
from bfs_mol import rollout_mol, draw_mol
import rdkit.Chem as Chem
import random
from Reinforcement_Learning.PPO.ppo_policy import ppo_policy
from graph_embedding import batch_from_smiles, batch_from_states

def one_hot_encode(indices, size):
    one_hot = torch.zeros(size, dtype = torch.float32)
    for i, idx in enumerate(indices):
        one_hot[i][idx] = 1.
    return one_hot

random.seed(1)

smiles = list(pd.read_csv('/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/Imitation_Learning/in-trials.csv')['smiles'])

initial_batch = batch_from_smiles([smiles[0]])
SURGE = ppo_policy(num_node_features = initial_batch.num_node_features)
optimizer = optim.Adam(SURGE.parameters(), lr = 1e-6)

epochs = 20
mini_batch_size = 128
for epoch in range(epochs): # Iterate through all the smiles
    random.shuffle(smiles)

    s_idx = 0
    total_bads = 0 # How many smiles strings cannot be parsed by RDKit
    while s_idx <= len(smiles): # Repeat until entire dataset is covered
        terminate = []
        atom1 = []
        atom2 = []
        bond = []
        states = []
        while len(terminate) < mini_batch_size: # Collect at least 128 steps of rollout
            try:
                t, a1, a2, b, s = rollout_mol(smiles[s_idx])
                s_idx += 1
            except Exception as e:
                s_idx += 1
                total_bads += 1
                continue

            print(f'{s_idx}/{len(terminate)}')

            terminate += t
            atom1 += a1
            atom2 += a2
            bond += b
            s = s[:-1] # Remove final state from list of states
            states += s

        # Shuffling rollout data
        combined_data = list(zip(terminate, atom1, atom2, bond, states))
        random.shuffle(combined_data)
        terminate, atom1, atom2, bond, states = zip(*combined_data)

        criterion = nn.CrossEntropyLoss()

        mini_batch_loss = 0
        batch = batch_from_states(states)
        t_logits, n_logits, b_logits = SURGE(batch.x, batch.edge_index, batch.batch)

        t_probs = t_logits.softmax(1)
        t_y = one_hot_encode(terminate, t_logits.size())
        t_loss = criterion(t_probs, t_y)

        # nmol
        nmol_probs = []
        state_idx, num_nodes = torch.unique(batch.batch, return_counts=True)
        for i in range(len(num_nodes)):  # Iterates through each graph in batch
            # Select first atom from existing molecule
            full_idx = sum(num_nodes[:i + 1])  # The length of the entire molecule and the atom bank
            prev_idx = sum(num_nodes[:i])

            if len(nmol_probs) == 0:
                nmol_probs = n_logits[prev_idx:full_idx - 10].softmax(0)
            else:
                nmol = n_logits[prev_idx:full_idx - 10].softmax(0)
                nmol_probs = torch.vstack((nmol_probs, nmol))
        num_nodes_nmol = [n - 10 for n in num_nodes]
        nmol_y = torch.zeros(nmol_probs.size(), dtype = torch.float32)
        for i in range(len(num_nodes_nmol)):
            prev_idx = sum(num_nodes_nmol[:i])
            nmol_y[prev_idx + atom1[i]] = 1.
        nmol_loss = criterion(nmol_probs, nmol_y)

        nfull_probs = []
        nfull_y = torch.zeros(n_logits.size(), dtype=torch.float32)
        for i in range(len(num_nodes)):
            full_idx = sum(num_nodes[:i + 1])
            prev_idx = sum(num_nodes[:i])
            nfull_y[prev_idx + atom2[i]] = 1.

            if len(nfull_probs) == 0:
                nfull_probs = n_logits[prev_idx:full_idx].softmax(0)
            else:
                nfull_probs = torch.vstack((nfull_probs, n_logits[prev_idx:full_idx].softmax(0)))
        nfull_loss = criterion(nfull_probs, nfull_y)

        b_probs = b_logits.softmax(1)
        b_y = one_hot_encode(bond, b_logits.size())
        b_loss = criterion(b_probs, b_y)

        optimizer.zero_grad()
        cumulative_loss = t_loss + nmol_loss + nfull_loss + b_loss
        print(f'    Cumulative loss: {cumulative_loss}')
        cumulative_loss.backward()
        optimizer.step()
