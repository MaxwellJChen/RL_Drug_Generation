import torch
from torch.optim import Adam
from Model.SURGE import SURGE
from Reinforcement_Learning.mol_env import vectorized_mol_env
from Model.graph_embedding import batch_from_states

import numpy as np

import wandb

"""
SURGE training with the REINFORCE algorithm. Intended for baseline testing and establishing
an efficient pipeline for more complex testing and architectures later.

The file can be split into 3 sections:
    1. Parameter Initialization & Configs
    2. Training Loop
    3. Loss Calculation & Gradient Ascent
"""

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

wandb.init(
    project = 'RL_Drug_Generation',
    name= f'Test',
    config={
        'lr': lr,
        'architecture': "CNN",
        'epochs': num_epochs,
        'gamma': gamma,
        'num_envs': num_envs
    })


# 2. Training Loop
for epoch in range(num_epochs):

    # Reset environment after each episode
    states = env.reset()

    # Episode loggers
    keys = ['t', 'nmol', 'nfull', 'b']
    saved_actions = {'t': [], 'nmol': [], 'nfull': [], 'b': []}
    saved_log_probs = {'t': [], 'nmol': [], 'nfull': [], 'b': []}
    saved_rewards = []

    # Episode computation
    for step in range(max_steps):

        # Compute actions and log probabilities
        batch = batch_from_states(states)
        actions, log_probs = model.act(batch, return_log_probs = True)

        # Record in episode loggers
        for key in keys:
            if step == 0:
                saved_actions[key] = actions[key]
                saved_log_probs[key] = log_probs[key]
            else:
                saved_actions[key] = np.vstack((saved_actions[key], actions[key]))
                saved_log_probs[key] = torch.vstack((saved_log_probs[key], log_probs[key]))

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
    all_loss = dict()
    cumulative_loss = 0
    for key in keys:

        # Find the average loss among vectorized environments for each SURGE component
        individual_loss = -1 * all_returns * saved_log_probs[key] / num_envs
        cumulative_loss += torch.sum(individual_loss)
        all_loss[key] = torch.sum(individual_loss)

    # Perform gradient ascent
    optimizer.zero_grad()
    cumulative_loss.backward()
    optimizer.step()

    print(cumulative_loss)

    wandb.log({"Cumulative Reward": cumulative_reward, "Cumulative Loss": cumulative_loss,
               "Termination Loss": all_loss['t'], "Nmol Loss": all_loss['nmol'],
               "Nfull Loss": all_loss['nfull'], "Bond Loss": all_loss['b']})

wandb.finish()