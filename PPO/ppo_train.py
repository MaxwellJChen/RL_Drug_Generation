import time
from collections import deque

import torch.optim as optim
import torch as torch

from ppo_policy import ppo_policy
from vectorized_mol_env import vectorized_mol_env
import copy

from graph_embedding import batch_from_states

"""
Proximal Policy Optimization with reference to https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
"""

# Initialize vectorized environment
num_envs = 2
mol_envs = vectorized_mol_env(max_mol_size = 10, max_steps = 100, num_envs = num_envs)
obs, info = mol_envs.reset()
initial_batch = batch_from_states(obs)

SURGE = ppo_policy(initial_batch.num_node_features)
act, log_prob, entropy = SURGE.act(initial_batch)

# Storage variables
all_obs = []
actions = []
log_probs = []
rewards = []
dones = []

# Rollout for 100 time steps
for i in range(100):
    # print(f'{i + 1} step')
    obs, reward, done, info = mol_envs.step(act[0], act[1], act[2], act[3])
    action, log_prob, entropy = SURGE.act(batch_from_states(obs))

    all_obs.append(obs)
    actions.append(action)
    log_probs.append(log_prob)
    rewards.append(reward)
    dones.append(done)

# Advantage estimation
gamma = 0.99
print(rewards)
# returns = deque(maxlen = len(rewards)) # Deque to use append left function
# for i in reversed(len(rewards)):
#     if i == len(rewards) - 1:
#         returns.appendleft(rewards[i])
#     else:
#         returns.appendleft(rewards[i] + gamma * returns[0])

"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_steps = 500 # Total timesteps

# Storage variables for rollout
# all_obs = deque(maxlen = num_steps)
# actions = deque(maxlen = num_steps)
# log_probs = deque(maxlen = num_steps)
# rewards = deque(maxlen = num_steps)
# dones = deque(maxlen = num_steps)
# values = deque(maxlen = num_steps)

global_step = 0

# Learning rate annealing
frac = 1.0 - (1 - 1.0) / num_steps
new_lr = frac * optimizer.param_groups[0]['lr']
optimizer.param_groups[0]['lr'] = new_lr

# Generating data from an episode
# for step in range(num_steps):
#     print(step)

    # all_obs.append(obs)
    # dones.append(done)

    # with torch.no_grad():
    #     action, log_prob, entropy = SURGE.act(batch_from_states(obs)) # Calculate action
    # actions.append(action)
    # log_probs.append(log_prob)

    # obs, reward, done, info = mol_envs.step(action[0], action[1], action[2], action[3]) # Step
"""