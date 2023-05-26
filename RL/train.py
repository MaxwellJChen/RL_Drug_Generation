from mol_env import Mol_Env
from policy import Policy
import graph_embedding as GE

import torch
import torch.optim as optim
import torch_geometric

import imageio

import rdkit.Chem as Chem
from rdkit.Chem import RWMol

from collections import deque

import numpy as np

"""
Caffeine: Cn1cnc2n(C)c(=O)n(C)c(=O)c12
Ibuprofen: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
Benzene: c1ccccc1
"""

def state_to_batch(state: RWMol):
    """
    Converts an RWMol into an embedded batch
    """

    batch = torch_geometric.data.Batch.from_data_list(GE.graph_from_smiles_atom_bank([Chem.MolToSmiles(state)]))
    return batch

def record_episode(policy, env):
    # Run a single episode
    terminated = False
    truncated = False
    num_steps = 0

    state, info = env.reset()  # Initial state
    p = policy
    frames = []
    frames.append(env.visualize(return_image=True))  # Visualize initial state
    while not (terminated or truncated):
        t_act, n_act, b_act = SURGE.act((state_to_batch(state)))
        state, reward, terminated, truncated, info = env.step(0, n_act[0][0], n_act[0][1], b_act[0])
        # print(f'Timestep: {info[0]} --- {"Valid" if not info[0] else "Invalid"} --- Termination: {t_act} --- Nodes: {n_act} --- Bond: {t_act}')
        frames.append(env.visualize(return_image=True))
    imageio.mimsave('episode.gif', frames, 'GIF', duration=1000 * 1 / 2)


# Sample graph
smiles = "c1ccccc1"
sample_graph = GE.single_graph_from_smiles_atom_bank(smiles)

# Model
SURGE_hyperparameters = {
    'gamma': 0.9,
    'lr': 0.0001
}
SURGE = Policy(num_node_features = sample_graph.num_node_features)
optimizer = optim.Adam(SURGE.parameters(), lr = SURGE_hyperparameters['lr'])

# Initialize environment
max_steps = 50
env = Mol_Env(max_mol_size = 50, max_steps = 50)
state, info = env.reset()

# REINFORCE
# Running a single episode
# Save actions and log probabilities for each "type" of action
t_log_probs = []
n1_log_probs = []
n2_log_probs = []
b_log_probs = []

rewards = []
terminated = False
truncated = False
while not (terminated or truncated):
    t_act, t_log_prob, n1_act, n1_log_prob, n2_act, n2_log_prob, b_act, b_log_prob = SURGE.act(state_to_batch(state))
    state, reward, terminated, truncated, info = env.step(1, n1_act, n2_act, b_act)

    # Saving results
    rewards.append(reward)
    t_log_probs.append(t_log_prob)
    n1_log_probs.append(n1_log_prob)
    n2_log_probs.append(n2_log_prob)
    b_log_probs.append(b_log_prob)

returns = deque(maxlen = max_steps)
n_steps = len(rewards)

for t in reversed(range(n_steps)):
    discounted_return = (returns[0] if len(returns) > 0 else 0)
    returns.appendleft(rewards[t] + SURGE_hyperparameters['gamma'] * discounted_return) # Efficiently calculated returns based on Bellman Equation

eps = np.finfo(np.float32).eps.item()
returns = torch.tensor(returns)
# Standardize returns
if len(returns) == 1:
    pass
else:
    returns = (returns - returns.mean())/(returns.std() + eps)

# Calculate loss of each action
t_loss = []
n1_loss = []
n2_loss = []
b_loss = []
cumulative_loss = []
for t, n1, n2, b, discounted_return in zip(t_log_probs, n1_log_probs, n2_log_probs, b_log_probs, returns):
    # print(t)
    # print(n1)
    # print(n2)
    # print(b)
    cumulative_loss.append((-t-n1-n2-b)*discounted_return)
    # t_loss.append(-t * discounted_return)
    # n1_loss.append(-n1 * discounted_return)
    # n2_loss.append(-n2 * discounted_return)
    # b_loss.append(-b * discounted_return)

print(cumulative_loss)
# print(t_loss)
# print(n1_loss)
# print(n2_loss)
# print(b_loss)
t_loss = t_loss.sum()
n1_loss = torch.cat(n1_loss).sum()
n2_loss = torch.cat(n2_loss).sum()
b_loss = torch.cat(b_loss).sum()
print(t_loss)
print(n1_loss)
print(n2_loss)
print(b_loss)

# Use PyTorch gradient descent to increase the predicted reward
optimizer.zero_grad()
cumulative_loss.backward()
optimizer.step()