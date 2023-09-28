import torch
from torch.optim import Adam
from SURGE import SURGE
from Reinforcement_Learning.mol_env import vectorized_mol_env
from graph_embedding import batch_from_states

import numpy as np

model = SURGE()
max_steps = 100
num_envs = 4
gamma = 0.99
eps = np.finfo(np.float32).eps.item()
env = vectorized_mol_env(num_envs = num_envs, max_steps = max_steps)
optimizer = Adam(params = model.parameters())

torch.autograd.set_detect_anomaly(True)

# Training loop
saved_actions = {'t': [], 'nmol': [], 'nfull': [], 'b': []}
saved_log_probs = {'t': [], 'nmol': [], 'nfull': [], 'b': []}
keys = ['t', 'nmol', 'nfull', 'b']
saved_rewards = []

num_epochs = 20
for epoch in range(num_epochs):
    states = env.reset()
    for step in range(max_steps):
        # Act
        batch = batch_from_states(states)
        actions, log_probs = model.act(batch, return_log_probs = True)
        for key in keys: # Record actions in dictionaries
            # No actions have been recorded. Multidimensional concatenation will return error.
            if step == 0:
                saved_actions[key] = actions[key]
                saved_log_probs[key] = log_probs[key]
            else:
                saved_actions[key] = np.vstack((saved_actions[key], actions[key]))
                saved_log_probs[key] = torch.vstack((saved_log_probs[key], log_probs[key]))

        # Step
        states, rewards, valids, timestep = env.step(actions['t'], actions['nmol'], actions['nfull'], actions['b'])
        if step == 0: # Record rewards
            saved_rewards = torch.tensor(rewards)
        else:
            saved_rewards = torch.vstack((saved_rewards, torch.tensor(rewards)))

    # Calculate returns
    saved_returns = torch.tensor(num_envs)
    returns = torch.zeros(num_envs)
    for idx in reversed(range(max_steps)):
        returns = saved_rewards[idx, :] + gamma * returns
        if idx == max_steps - 1:
            saved_returns = returns
        else:
            saved_returns = torch.vstack((returns, saved_returns))
    saved_returns = (saved_returns - saved_returns.mean()) / (saved_returns.std() + eps)

    # Calculate loss
    cumulative_loss = 0
    for key in keys:
        individual_loss = -1 * saved_returns * saved_log_probs[key]
        cumulative_loss += torch.sum(individual_loss)

    # Perform gradient ascent
    optimizer.zero_grad()
    cumulative_loss.backward()
    optimizer.step()

    print(cumulative_loss)