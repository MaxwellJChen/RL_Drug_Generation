import torch.optim as optim

import pandas as pd
import random

from rollout import rollout
from SURGE import SURGE
from graph_embedding import batch_from_smiles

random.seed(1)

"""
Pseudocode:
    1. Initialize variables
    2. Get minibatches of the right size
    3. Obtain SURGE probability distributions
    4. Formulate true probability distributions
    5. Obtain crossentropy loss
    6. Update model
"""

"""Initialize Variables"""
smiles = list(pd.read_csv('/Data/in_trials_filtered.csv')['smiles'])
valences = {'C': 4, 'O': 2, 'N': 3, 'S': 6, 'F': 1, 'Cl': 1, 'P': 5, 'Br': 1, 'I': 1, 'B': 3}

# Specify model and learning algorithm parameters
sample = batch_from_smiles([smiles[0]])
SURGE = SURGE(num_node_features = sample.num_node_features)
lr = 1e-6
optimizer = optim.Adam(SURGE.parameters(), lr = lr)

record_every = 100 # How many trials until recording a video of model performance

# Training loop parameters
epochs = 20
minibatch_size = 128
minibatch = {'terminate': [], 'nmol': [], 'nfull': [], 'bond': [], 'state': []}
minibatch_extra = {'terminate': [], 'nmol': [], 'nfull': [], 'bond': [], 'state': []}
keys = ['terminate', 'nmol', 'nfull', 'bond', 'state']

smiles = smiles[:10]

"""Get Minibatches"""
i = 0
while i < len(smiles):
    while len(minibatch['terminate']) < minibatch_size:  # If size of minibatch surpasses minibatch size after adding rollout steps of a molecule, stop decomposing more molecules
        if i == len(smiles): # If we reach the end of the dataset before the minibatch contains 128 steps
            break
        steps = rollout(smiles[i]) # Converting SMILES strings to rollout steps
        for j, key in enumerate(keys): # Adding the rollout steps to the minibatch dictionary
            minibatch[key] += steps[j]
        i += 1 # Moving on to the next SMILES string

    if len(minibatch['terminate']) > minibatch_size: # If minibatch size is correct, empty minibatch_extra is correct
        for key in keys:
            minibatch_extra[key] = minibatch[key][128:] # Adding extra rollout steps to minibatch_extra
            minibatch[key] = minibatch[key][:128] # Indexing off extra rollout steps from minibatch

    """SURGE Probability Distributions"""
    SURGE()

    for key in keys:
        minibatch[key] = minibatch_extra[key]
        minibatch_extra[key] = []

print('Training done!')