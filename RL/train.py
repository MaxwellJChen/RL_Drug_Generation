from mol_env import Mol_Env
from policy import Mol_GCN
import graph_embedding as GE

import torch
from torch.distributions import Categorical
import torch_geometric

import rdkit.Chem as Chem

# Initialize environment
mol_env = Mol_Env(max_mol_size = 50, max_steps = 50)
state, info = mol_env.reset() # Initial observation

# Sample graph
smiles = "c1ccccc1"
sample_graph = GE.graph_from_smiles_atom_bank([smiles])[0]

# Model
gcn = Mol_GCN(num_node_features = sample_graph.num_node_features)

# First action

done = False
num_steps = 0
while not done:
    num_steps += 1
    # print(num_steps)
    c_bond = Categorical(b.view(-1))
    state, reward, terminated, truncated, info = mol_env.step(0, selected_idx[0][0], selected_idx[0][1], c_bond.sample())
    print(reward)
    done = terminated or truncated

    g, num_nodes = ge.graph_from_smiles_atom_bank_with_list([Chem.MolToSmiles(state)])
    g = torch_geometric.data.Batch.from_data_list(g)
    t, selected_idx, b = SURGE(g.x, g.edge_index, num_nodes, g.batch)
    # print(selected_idx)
    mol_env.visualize(info[0])