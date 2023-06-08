import numpy as np

from ppo_policy import ppo_policy
from mol_env import single_mol_env

from graph_embedding import batch_from_states, batch_from_smiles

import torch
from torch.distributions.categorical import Categorical

import random
import copy
import matplotlib.pyplot as plt
import rdkit.Chem.Draw as Draw

"""
Proximal Policy Optimization with reference to https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/, https://www.youtube.com/watch?v=HR8kQMTO8bk&t=5s, and https://colab.research.google.com/drive/1MsRlEWRAk712AQPmoM9X9E6bNeHULRDb?usp=sharing
"""

mol_env = single_mol_env(max_mol_size = 10, max_steps = 100)
obs, info = mol_env.reset()
initial_batch = batch_from_states([obs])

SURGE = ppo_policy(initial_batch.num_node_features)
lr = 1e-5
optimizer = torch.optim.Adam(SURGE.parameters(), lr, eps = 1e-5)

num_episodes = 100
num_rollout_steps = 100
max_iters = 100
for episode in range(num_episodes): # How many times to execute rollout
    # Learning rate annealing
    frac = 1.0 - (episode) / num_episodes
    optimizer.param_groups[0]['lr'] = frac * lr

    # Storage variables
    all_obs = []
    rewards = []
    total_episode_reward = 0

    t_acts = []
    old_t_log_probs = []
    nmol_acts = []
    old_nmol_log_probs = []
    nfull_acts = []
    old_nfull_log_probs = []
    b_acts = []
    old_b_log_probs = []

    # Executing rollout
    for _ in range(num_rollout_steps):
        with torch.no_grad():
            t_act, t_log_prob, t_entropy, n1_act, n2_act, nmol_act, nmol_log_prob, nmol_entropy, nfull_act, nfull_log_prob, nfull_entropy, b_act, b_log_prob, b_entropy = SURGE.act(batch_from_states([obs]), mol_env)
        next_obs, reward, terminated, truncated, info = mol_env.step(t_act[0], n1_act[0], n2_act[0], b_act[0])
        if terminated or truncated:
            next_obs, info = mol_env.reset()

        all_obs.append(obs)
        obs = next_obs
        rewards.append(reward)
        total_episode_reward += reward

        t_acts.append(t_act)
        old_t_log_probs.append(t_log_prob)
        nmol_acts.append(nmol_act)
        old_nmol_log_probs.append(nmol_log_prob)
        nfull_acts.append(nfull_act)
        old_nfull_log_probs.append(nfull_log_prob)
        b_acts.append(b_act)
        old_b_log_probs.append(b_log_prob)

    print(f'Episode {episode + 1} reward: {total_episode_reward}')

    # Calculate returns
    gamma = 0.99
    returns = [float(rewards[-1])]
    for i in reversed(range(len(rewards) - 1)):
        returns.append(float(rewards[i]) + returns[-1] * gamma)
    returns = returns[::-1]
    returns = (returns - np.mean(returns)) / np.std(returns) # Normalize returns
    # Cannot calculate GAEs without value estimate, so using normalized returns instead

    # Shuffle the rollout data
    combined_data = list(zip(all_obs, rewards, t_acts, old_t_log_probs, nmol_acts, old_nmol_log_probs, nfull_acts, old_nfull_log_probs, b_acts, old_b_log_probs))
    random.shuffle(combined_data)
    all_obs, rewards, t_acts, old_t_log_probs, nmol_acts, old_nmol_log_probs, nfull_acts, old_nfull_log_probs, b_acts, old_b_log_probs = zip(*combined_data)

    # PPO training
    for _ in range(max_iters): # How many times to experience replay over a single run of rollout data
        # print(f'    {_} iter')

        optimizer.zero_grad()

        # Calculate new log probabilities
        new_t_log_probs = []
        new_nmol_log_probs = []
        new_nfull_log_probs = []
        new_b_log_probs = []

        ep_obs_batch = batch_from_states(all_obs)
        t, n, b = SURGE(ep_obs_batch.x, ep_obs_batch.edge_index, ep_obs_batch.batch)

        t_categorical = Categorical(logits = t)
        new_t_log_probs = t_categorical.log_prob(torch.tensor([t[0] for t in t_acts], dtype = torch.float32))
        t_entropy = t_categorical.entropy().mean()

        state_idx, num_nodes = torch.unique(ep_obs_batch.batch, return_counts=True)
        for i in range(len(num_nodes)): # Iterating through the number of molecules in a batch
            full_idx = sum(num_nodes[:i + 1]) # The length of the entire molecule and the atom bank
            prev_idx = sum(num_nodes[:i])

            prob_molecule = n[prev_idx:full_idx - 10].view(-1)

            # Masking

            prob_molecule = prob_molecule.softmax(dim=0)
            nmol_categorical = Categorical(prob_molecule)
            new_nmol_log_probs.append(nmol_categorical.log_prob(nmol_acts[i][0]))
            nmol_entropy = nmol_categorical.entropy().mean()

            prob_full = n[prev_idx:full_idx].view(-1)
            prob_full = prob_full.softmax(dim=0)
            nfull_categorical = Categorical(prob_full)
            new_nfull_log_probs.append(nfull_categorical.log_prob(nfull_acts[i][0]))
            nfull_entropy = nfull_categorical.entropy().mean()

        b_categorical = Categorical(logits = b)
        new_b_log_probs = b_categorical.log_prob(torch.tensor([b[0] for b in b_acts], dtype = torch.float32))
        b_entropy = b_categorical.entropy().mean()

        # Calculate ratios and loss for each action in multidiscrete environment
        ppo_clip_epsilon = 0.2
        # Termination
        t_log_diffs = torch.stack([new-old[0] for new, old in zip(new_t_log_probs, old_t_log_probs)])
        t_full_ratios = torch.exp(t_log_diffs)
        t_clipped_ratios = t_full_ratios.clamp(1 - ppo_clip_epsilon, 1 + ppo_clip_epsilon)
        t_full_loss = torch.stack([t_full_ratio * r for t_full_ratio, r in zip(t_full_ratios, returns)])
        t_clipped_loss = torch.stack([t_clipped_ratio * r for t_clipped_ratio, r in zip(t_clipped_ratios, returns)])
        t_loss = -torch.min(t_full_loss, t_clipped_loss).mean()

        # Molecule only
        nmol_log_diffs = torch.stack([new - old[0] for new, old in zip(new_nmol_log_probs, old_nmol_log_probs)])
        nmol_full_ratios = torch.exp(nmol_log_diffs)
        nmol_clipped_ratios = nmol_full_ratios.clamp(1 - ppo_clip_epsilon, 1 + ppo_clip_epsilon)
        nmol_full_loss = torch.stack([nmol_full_ratio * r for nmol_full_ratio, r in zip(nmol_full_ratios, returns)])
        nmol_clipped_loss = torch.stack([nmol_clipped_ratio * r for nmol_clipped_ratio, r in zip(nmol_clipped_ratios, returns)])
        nmol_loss = -torch.min(nmol_full_loss, nmol_clipped_loss).mean()

        # Molecule and atom bank
        nfull_log_diffs = torch.stack([new - old[0] for new, old in zip(new_nfull_log_probs, old_nfull_log_probs)])
        nfull_full_ratios = torch.exp(nfull_log_diffs)
        nfull_clipped_ratios = nfull_full_ratios.clamp(1 - ppo_clip_epsilon, 1 + ppo_clip_epsilon)
        nfull_full_loss = torch.stack([nfull_full_ratio * r for nfull_full_ratio, r in zip(nfull_full_ratios, returns)])
        nfull_clipped_loss = torch.stack([nfull_clipped_ratio * r for nfull_clipped_ratio, r in zip(nfull_clipped_ratios, returns)])
        nfull_loss = -torch.min(nfull_full_loss, nfull_clipped_loss).mean()

        # Bond
        b_log_diffs = torch.stack([new-old[0] for new, old in zip(new_b_log_probs, old_b_log_probs)])
        b_full_ratios = torch.exp(b_log_diffs)
        b_clipped_ratios = b_full_ratios.clamp(1 - ppo_clip_epsilon, 1 + ppo_clip_epsilon)
        b_full_loss = torch.stack([b_full_ratio * r for b_full_ratio, r in zip(b_full_ratios, returns)])
        b_clipped_loss = torch.stack([b_clipped_ratio * r for b_clipped_ratio, r in zip(b_clipped_ratios, returns)])
        b_loss = -torch.min(b_full_loss, b_clipped_loss).mean()

        # Backpropagation
        # print(t_loss)
        # print(nmol_loss)
        # print(nfull_loss)
        # print(b_loss)
        cumulative_loss = t_loss + nmol_loss + nfull_loss + b_loss
        # print(cumulative_loss)
        cumulative_loss.backward()
        optimizer.step()

        # KL divergence check
        target_kl_div = 0.01
        if t_log_diffs.mean() > target_kl_div or nmol_log_diffs.mean() > target_kl_div or nfull_log_diffs.mean() > target_kl_div or b_log_diffs.mean() > target_kl_div:
            break