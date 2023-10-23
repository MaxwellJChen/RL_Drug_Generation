import numpy as np

from Model.masked_model import SURGE
from Reinforcement_Learning.mol_env import mol_env

from Model.graph_embedding import batch_from_states

import torch

"""
Proximal Policy Optimization with reference to https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/, https://www.youtube.com/watch?v=HR8kQMTO8bk&t=5s, and https://colab.research.google.com/drive/1MsRlEWRAk712AQPmoM9X9E6bNeHULRDb?usp=sharing
"""


# def episode(policy):
#     """
#     Records an episode with a single state in the mol_env to assess model performance.
#     """
#     mol_env = mol_env(num_envs = 1)

SURGE = SURGE()

num_envs = 4
mol_env = mol_env(num_envs = num_envs)

all_states = []
all_rewards = []

t_acts = []
t_log_probs = []
t_masks = []

nmol_acts = []
nmol_log_probs = []
all_nmol_masks = []

nfull_acts = []
nfull_log_probs = []
all_nfull_masks = []

b_acts = []
b_log_probs = []
b_masks = []

values = []

# Rollout
states = mol_env.reset()
for _ in range(3):
    batch = batch_from_states(states)
    with torch.no_grad():
        t_act, t_log_prob, t_mask, nmol_act, nmol_log_prob, nmol_masks, nfull_act, nfull_log_prob, nfull_masks, b_act, b_log_prob, b_mask = SURGE.act(batch, states)
        value = SURGE.value(batch)

    all_states.append(states)

    states, rewards, valids, timestep = mol_env.step(t_act, nmol_act, nfull_act, b_act)

    all_rewards.append(rewards)

    t_acts.append(t_act)
    t_log_probs.append(t_log_prob)
    t_masks.append(t_mask)

    nmol_acts.append(nmol_act)
    nmol_log_probs.append(nmol_log_prob)
    all_nmol_masks.append(nmol_masks)

    nfull_acts.append(nfull_act)
    nfull_log_probs.append(nfull_log_prob)
    all_nfull_masks.append(nfull_masks)

    b_acts.append(b_act)
    b_log_probs.append(b_log_prob)
    b_masks.append(b_mask)

    values.append(value)

# Calculate returns
gamma = 0.99
all_returns = []
print(all_rewards)
for i in range(len(all_rewards)):
    returns = [float(all_rewards[i][-1])]
    for j in reversed(range(len(all_rewards[i]) - 1)):
        returns.append(float(all_rewards[i][j]) + returns[-1] * gamma)
    returns = returns[::-1]
    returns = (returns - np.mean(returns)) / np.std(returns)
    print(returns)

# Calculate GAEs

# Compute new log_probs

# 