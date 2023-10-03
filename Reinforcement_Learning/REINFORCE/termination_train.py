import torch
from torch.optim import Adam
from SURGE import SURGE
from Reinforcement_Learning.mol_env import vectorized_mol_env
from graph_embedding import batch_from_states

import numpy as np

import wandb

"""Training termination model first for curriculum learning effect."""

# 1. Parameter Initialization & Configs
model = SURGE()
max_steps = 100
num_envs = 4
env = vectorized_mol_env(num_envs = num_envs, max_steps = max_steps) # Vectorized molecular environment
lr = 0.02
optimizer = Adam(lr = lr, params = model.parameters())

gamma = 0.99
eps = np.finfo(np.float32).eps.item() # Small constant to decrease numerical instability
num_epochs = 20

# wandb.init(
#     project = 'RL_Drug_Generation',
#     name= f'Termination_Test',
#     config={
#         'lr': lr,
#         'epochs': num_epochs,
#         'gamma': gamma,
#         'num_envs': num_envs
#     })


# 2. Training Loop
for epoch in range(num_epochs):

    # Reset environment after each episode
    states = env.reset()

    # Episode loggers
    keys = ['t']
    saved_actions = []
    saved_log_probs = []
    saved_rewards = []

    # Episode computation
    for step in range(max_steps):

        # Compute actions and log probabilities
        batch = batch_from_states(states)
        actions, log_probs = model.act(batch, return_log_probs = True)

        # Record in episode loggers
        if step == 0:
            saved_actions = actions['t']
            saved_log_probs = log_probs['t']
        else:
            saved_actions = np.vstack((saved_actions, actions['t']))
            saved_log_probs = torch.vstack((saved_log_probs, log_probs['t']))

        # Take a step in environment
        states, rewards, valids, timestep = env.step(actions['t'], actions['nmol'], actions['nfull'], actions['b'])

        # Record rewards
        if step == 0:
            saved_rewards = torch.tensor(rewards)
        else:
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

    # Perform gradient ascent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)

    # wandb.log({"Cumulative Reward": cumulative_reward, "Termination Loss": loss})

# wandb.finish()