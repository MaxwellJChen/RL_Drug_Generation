from mol_env import single_mol_env
from reinforce_policy import reinforce_policy
from graph_embedding import batch_from_smiles

import torch
import torch.optim as optim

import imageio

import rdkit.Chem as Chem

from collections import deque

import numpy as np

def record_episode(policy, env):
    """
    Runs a single full episode with a policy (trained or untrained) and an environment. Records as a GIF file.
    """

    # Run a single episode
    terminated = False
    truncated = False
    num_steps = 0

    state, info = env.reset()  # Initial state
    p = policy
    frames = []
    frames.append(env.visualize(return_image=True))  # Visualize initial state
    while not (terminated or truncated):
        t_act, t_log_prob, n1_act, n1_log_prob, n2_act, n2_log_prob, b_act, b_log_prob = SURGE.act(batch_from_smiles([Chem.MolToSmiles(state)]))
        state, reward, terminated, truncated, info = env.step(t_act, n1_act, n2_act, b_act)
        # print(f'Timestep: {info[0]} --- {"Valid" if not info[0] else "Invalid"} --- Termination: {t_act} --- Nodes: {n1_act} & {n2_act} --- Bond: {b_act}')
        frames.append(env.visualize(return_image=True))
    imageio.mimsave('episode.gif', frames, 'GIF', duration = 1000/5)

# Sample graph
smiles = "c1ccccc1"
sample_batch = batch_from_smiles([smiles])

# Model
SURGE_hyperparameters = {
    'gamma': 0.9,
    'lr': 0.0001,
    'num_episodes': 10,
}
SURGE = reinforce_policy(num_node_features = sample_batch.num_node_features)
optimizer = optim.Adam(SURGE.parameters(), lr = SURGE_hyperparameters['lr'])

# Initialize environment
max_steps = 100
env = single_mol_env(max_mol_size=70, max_steps=max_steps)

# REINFORCE
print_every = 50 # How many episodes before printing mean and std of SURGE scores
eps_scores = []
avg_scores = [] # Tracks the mean scores over time
for i in range(SURGE_hyperparameters['num_episodes']):
    print(f'Episode {i + 1}')
    # Running a single episode
    state, info = env.reset()

    # Save actions and log probabilities for each "type" of action
    t_log_probs = []
    n1_log_probs = []
    n2_log_probs = []
    b_log_probs = []

    rewards = []
    terminated = False
    truncated = False

    while not (terminated or truncated): # Stepping through a new episode
        t_act, t_log_prob, n1_act, n1_log_prob, n2_act, n2_log_prob, b_act, b_log_prob = SURGE.act(batch_from_smiles([Chem.MolToSmiles(state)]))
        state, reward, terminated, truncated, info = env.step(t_act, n1_act, n2_act, b_act)

        # Recording results
        rewards.append(reward)
        t_log_probs.append(t_log_prob)
        n1_log_probs.append(n1_log_prob)
        n2_log_probs.append(n2_log_prob)
        b_log_probs.append(b_log_prob)

    eps_scores.append(sum(rewards))

    returns = deque(maxlen = max_steps)
    n_steps = len(rewards)

    for t in reversed(range(n_steps)):
        discounted_return = (returns[0] if len(returns) > 0 else 0)
        returns.appendleft(rewards[t] + SURGE_hyperparameters['gamma'] * discounted_return) # Efficiently calculated returns based on Bellman Equation

    eps = np.finfo(np.float32).eps.item()
    returns = torch.tensor(returns)
    # Standardize returns
    if len(returns) != 1:
        returns = (returns - returns.mean())/(returns.std() + eps)

    # Calculate a cumulative loss by combining the losses of each action
    cumulative_loss = []
    for t, n1, n2, b, discounted_return in zip(t_log_probs, n1_log_probs, n2_log_probs, b_log_probs, returns):
        cumulative_loss.append((-t-n1-n2-b)*discounted_return)

    # Sum the losses
    if len(cumulative_loss) != 1:
        cumulative_loss = torch.cat(cumulative_loss).sum()
    else:
        cumulative_loss = cumulative_loss[0]

    # Use PyTorch gradient descent to increase the predicted reward
    optimizer.zero_grad()
    cumulative_loss.backward()
    optimizer.step()

    if (i+1) % print_every == 0:
        print(f'Episode {i+1} --- Average Score: {np.mean(eps_scores)} --- STD Scores: {np.std(eps_scores)}')
        avg_scores.append(np.mean(eps_scores))
        eps_scores = []

state, info = env.reset()
# record_episode(SURGE, env)