import torch
from torch.optim import Adam
from Model.SURGE import SURGE
from Reinforcement_Learning.mol_env import vectorized_mol_env
from Model.graph_embedding import batch_from_states

import numpy as np

import wandb

"""Training termination model first for curriculum learning effect."""

# 1. Parameter Initialization & Configs
model = SURGE()
max_steps = 200
num_envs = 4
env = vectorized_mol_env(num_envs = num_envs, max_steps = max_steps) # Vectorized molecular environment
lr = 0.02
optimizer = Adam(lr = lr, params = model.parameters())

gamma = 0.99
eps = np.finfo(np.float32).eps.item() # Small constant to decrease numerical instability
num_episodes = 100

wandb.init(
    project = 'RL_Drug_Generation',
    name= f'Termination_Test',
    config={
        'lr': lr,
        'episodes': num_episodes,
        'gamma': gamma,
        'num_envs': num_envs
    })


# 2. Training Loop
for epoch in range(num_episodes):

    # Reset environment after each episode
    states = env.reset()

    # Episode loggers
    saved_actions = []
    saved_log_probs = []
    saved_rewards = []
    saved_mol_sizes = []

    # Episode computation
    for step in range(max_steps):

        # Compute actions and log probabilities
        batch = batch_from_states(states)
        actions, log_probs = model.act(batch, return_log_probs = True)

        # Take a step in environment
        states, rewards, valids, timestep = env.step(actions['t'], actions['nmol'], actions['nfull'], actions['b'], version = 't')

        # Record in episode loggers
        if step == 0:
            saved_mol_sizes = env.mol_sizes
            saved_actions = actions['t']
            saved_log_probs = log_probs['t']
            saved_rewards = torch.tensor(rewards)
        else:
            saved_mol_sizes = np.vstack((saved_mol_sizes, env.mol_sizes))
            saved_actions = np.vstack((saved_actions, actions['t']))
            saved_log_probs = torch.vstack((saved_log_probs, log_probs['t']))
            saved_rewards = torch.vstack((saved_rewards, torch.tensor(rewards)))

    # 3. Loss Calculation & Gradient Ascent
    cumulative_reward = torch.sum(saved_rewards) / num_envs

    # Calculate returns
    all_returns = torch.tensor(num_envs)
    returns = torch.zeros(num_envs)
    for idx in reversed(range(max_steps)):
        returns = saved_rewards[idx, :] + gamma * returns
        if idx == max_steps - 1:
            all_returns = returns
        else:
            all_returns = torch.vstack((returns, all_returns))
    all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + eps)

    # Calculate loss
    loss = torch.sum(-1 * all_returns * saved_log_probs / num_envs)

    # Perform gradient ascent on cumulative reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    wandb.log({"Cumulative Reward": cumulative_reward, "Loss": loss, "Average Size": saved_mol_sizes.mean()})

wandb.finish()