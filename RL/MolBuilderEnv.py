import gymnasium as gym
from gymnasium import spaces
import numpy as np

import torch
import torch_geometric
from torch_geometric.data import Data
from graph_embedding import single_bare_graph_from_smiles

class MolBuilderEnv(gym.Env):
    """
    Environment to train PyG RL agent
    Observations: the current molecular graph
    Actions: picking two of all the atom indices and a bond type
    Done: terminates either when the agent decides to or when the molecule surpasses 70 heavy atoms
    Reward: scalar value based on QED, validity, and SAS of molecule
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Main functions
    def __init__(self):
        self.molecule = single_bare_graph_from_smiles("CC") # Always starts with a single C-C single bond

        self.node_bank = torch.zeros([12, 12], dtype = torch.float32)
        for i in range(12):
            self.node_bank[i][i] = 1.0


        self.action_space = spaces.Box(low = 0, high = 80, dtype = np.int64, shape = 3) # node1, node2, and bond type. Max of 70 nodes in a graph and 12 from node bank
        self.observation_space = spaces.Graph(node_space=spaces.Box(low=0, high=1, dtype= np.int64, shape=(12,)),
                                              edge_space=spaces.Box(low=0, high=1, dtype = np.int64, shape=(4,)), seed=42) # Bare molecular embeddings

        # Auxiliary functions
    def add_node(self, g: Data, node_embedding: torch.Tensor) -> Data:
        """Adds a new node to the very end of the node feature vector"""
        g.x = torch.vstack((g.x, node_embedding))
        return g

    def add_edge(self, g: Data, node1: int, node2: int, edge_embedding: torch.Tensor) -> Data:
        """Adds a new edge to a molecule"""

        edge_index = g.edge_index.numpy()
        edge_attr = g.edge_attr.numpy()

        if node1 > node2:
            placeholder = node2
            node2 = node1
            node1 = placeholder

        # Updating edge index
        idx1 = np.max(np.where(edge_index[0] == node1)[0]) + 1
        edge_index = np.array([np.insert(edge_index[0], idx1, node1).tolist(),
                               np.insert(edge_index[1], idx1, node2).tolist()])

        if node2 >= g.num_nodes - 1:  # If the node index is the final node (which might not be connected to anything yet), add to end
            edge_index = np.array([edge_index[0].tolist() + [node2],
                                   edge_index[1].tolist() + [node1]])
        else:
            idx2 = np.max(np.where(edge_index[0] == node2)[0]) + 1
            edge_index = np.array([np.insert(edge_index[0], idx2, node2).tolist(),
                                   np.insert(edge_index[1], idx2, node1).tolist()])

        edge_index = torch.tensor(edge_index, dtype=torch.float32)
        g.edge_index = edge_index

        # Updating edge attr
        edge_attr = np.insert(edge_attr, idx1, edge_embedding, axis=0)
        edge_attr = np.insert(edge_attr, idx2, edge_embedding, axis=0)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        g.edge_attr = edge_attr

        return g



    def _get_info(self):
        pass

    def reset(self):
        self.molecule = single_bare_graph_from_smiles("CC") # Resets environment to C-C bond
        return self.molecule, 2 # Returns the molecule observation and a scalar representing how many heavy atoms there are

    def step(self, action):
        # Identify embeddings for selected nodes
        atom_attr = torch.vstack((self.molecule.x, self.node_bank))

        # If a new atom (node2) is being added on to the original molecule
        if action[1] > (self.molecule.num_nodes-1):
            node2 = atom_attr[action[1], :] # Get embedding for new atom
            edge = torch.zeros(4, dtype = torch.float32)
            edge[action[2] - 1] = 1.

            self.molecule = self.add_node(self.molecule, node2)
            self.molecule = self.add_edge(self.molecule, action[0], self.molecule.num_nodes - 1, edge)


        # If only a new bond is being created
        # else:
        pass

    def render(self):
        pass