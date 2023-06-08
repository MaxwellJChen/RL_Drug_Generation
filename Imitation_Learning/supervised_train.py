import torch
import torch.optim as optim

import pandas as pd
from bfs_mol import rollout_mol, draw_mol
import rdkit.Chem as Chem
import random
from PPO.ppo_policy import ppo_policy
from graph_embedding import batch_from_smiles, batch_from_states

random.seed(42)

smiles = list(pd.read_csv('/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/Imitation_Learning/in-trials.csv')['smiles'])

initial_batch = batch_from_smiles([smiles[0]])
SURGE = ppo_policy(num_node_features = initial_batch.num_node_features)
optimizer = optim.Adam(SURGE.parameters(), lr = 1e-6, eps = 1e-8)

# for _ in range(len(smiles)):
terminate, atom1, atom2, bond, states = rollout_mol(smiles[0])

# Shuffling rollout data
states = states[:-1] # Remove the final state in states for shuffling
combined_data = list(zip(terminate, atom1, atom2, bond, states))
random.shuffle(combined_data)
terminate, atom1, atom2, bond, states = zip(*combined_data)

t_distribution, n_distribution, b_distribution = SURGE(states)
print(t_distribution)

for t, a1, a2, b in zip(terminate, atom1, atom2, bond):
    pass