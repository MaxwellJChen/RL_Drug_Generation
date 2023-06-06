import numpy as np

from ppo_policy import ppo_policy
from mol_env import vectorized_mol_env
from mol_env import single_mol_env

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

# Storage variables
all_obs = []
actions = []
log_probs = []
rewards = []
dones = []
ep_reward = [0, 0]

# Rollout for 100 time steps
for i in range(3):
    # print(f'{i + 1} step')
    action, log_prob, entropy = SURGE.act(batch_from_states(obs))
    next_obs, reward, done, info = mol_envs.step(action[0], action[1], action[2], action[3])


    all_obs.append(obs)
    actions.append(action)
    log_probs.append(log_prob)
    rewards.append(reward)
    dones.append(done)

    obs = next_obs

    ep_reward = [ep_rew + rew for ep_rew, rew in zip(ep_reward, reward)]

# Calculate returns
gamma = 0.99
returns = [float(rewards[-1])]
for i in reversed(range(len(rewards) - 1)):
    returns.append(float(rewards[i]) + returns[-1] * gamma)

# Calculate GAEs
next_values = np.concatenate([returns[1:], [0]])
deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, returns, next_values)]

gamma = 0.99