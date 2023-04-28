# Gymnasium
import gymnasium as gym
from gymnasium import spaces

# Torch
import torch
import torch_geometric
from torch_geometric.data import Data

import numpy as np

# RDKit
from rdkit import Chem
import rdkit.Chem.QED as QED

class MolBuilderEnv(gym.Env):
    """
    Environment to train PyG RL agent
    Observations: the current molecular graph
    Actions: picking two of all the atom indices and a bond type
    Done: terminates either when the agent decides to or when the molecule surpasses 70 heavy atoms
    Reward: scalar value based on QED, validity, and SAS of molecule
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

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

    def graph_to_smiles(self, data: Data, kekulize: bool = False):
        """
        https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/smiles.html
        Converts a :class:`torch_geometric.data.Data` instance to a SMILES
        string.

        Args:
            data (torch_geometric.data.Data): The molecular graph.
            kekulize (bool, optional): If set to :obj:`True`, converts aromatic
                bonds to single/double bonds. (default: :obj:`False`)
        """

        x_map = {
            'atomic_num':
                list(range(0, 119)),
            'chirality': [
                'CHI_UNSPECIFIED',
                'CHI_TETRAHEDRAL_CW',
                'CHI_TETRAHEDRAL_CCW',
                'CHI_OTHER',
                'CHI_TETRAHEDRAL',
                'CHI_ALLENE',
                'CHI_SQUAREPLANAR',
                'CHI_TRIGONALBIPYRAMIDAL',
                'CHI_OCTAHEDRAL',
            ],
            'degree':
                list(range(0, 11)),
            'formal_charge':
                list(range(-5, 7)),
            'num_hs':
                list(range(0, 9)),
            'num_radical_electrons':
                list(range(0, 5)),
            'hybridization': [
                'UNSPECIFIED',
                'S',
                'SP',
                'SP2',
                'SP3',
                'SP3D',
                'SP3D2',
                'OTHER',
            ],
            'is_aromatic': [False, True],
            'is_in_ring': [False, True],
        }

        mol = Chem.RWMol()

        for i in range(data.num_nodes):
            atom = Chem.Atom(data.x[i, 0].item())
            atom.SetChiralTag(Chem.rdchem.ChiralType.values[data.x[i, 1].item()])
            atom.SetFormalCharge(x_map['formal_charge'][data.x[i, 3].item()])
            atom.SetNumExplicitHs(x_map['num_hs'][data.x[i, 4].item()])
            atom.SetNumRadicalElectrons(
                x_map['num_radical_electrons'][data.x[i, 5].item()])
            atom.SetHybridization(
                Chem.rdchem.HybridizationType.values[data.x[i, 6].item()])
            atom.SetIsAromatic(data.x[i, 7].item())
            mol.AddAtom(atom)

        edges = [tuple(i) for i in data.edge_index.t().tolist()]
        visited = set()

        for i in range(len(edges)):
            src, dst = edges[i]
            if tuple(sorted(edges[i])) in visited:
                continue

            bond_type = Chem.BondType.values[data.edge_attr[i, 0].item()]
            mol.AddBond(src, dst, bond_type)

            # Set stereochemistry:
            stereo = Chem.rdchem.BondStereo.values[data.edge_attr[i, 1].item()]
            if stereo != Chem.rdchem.BondStereo.STEREONONE:
                db = mol.GetBondBetweenAtoms(src, dst)
                db.SetStereoAtoms(dst, src)
                db.SetStereo(stereo)

            # Set conjugation:
            is_conjugated = bool(data.edge_attr[i, 2].item())
            mol.GetBondBetweenAtoms(src, dst).SetIsConjugated(is_conjugated)

            visited.add(tuple(sorted(edges[i])))

        mol = mol.GetMol()

        if kekulize:
            Chem.Kekulize(mol)

        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol)

        return Chem.MolToSmiles(mol, isomericSmiles=True)

    def one_hot_encoding(self, x, permitted_list):
        """
        Maps input elements x which are not in the permitted list to the last element of the permitted list.
        """
        if x not in permitted_list:
            x = permitted_list[-1]
        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
        return binary_encoding

    def graph_from_smiles(self, smiles: str) -> torch_geometric.data.Data:
        """
        Creates one graph from one smiles string with only atom type one-hot-encoding
        """

        atomic_symbols = ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B', 'Si']
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        n_node_features = 11
        n_edge_features = 4

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = self.one_hot_encoding(atom.GetSymbol(), atomic_symbols)

        X = torch.tensor(X, dtype=torch.float)

        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(Chem.GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))

        for (k, (i, j)) in enumerate(zip(rows, cols)):
            EF[k] = self.one_hot_encoding(mol.GetBondBetweenAtoms(int(i), int(j)).GetBondType(),
                                          bond_types)

        EF = torch.tensor(EF, dtype=torch.float)

        # construct Pytorch Geometric data object
        return Data(x=X, edge_index=E, edge_attr=EF)

    # Main functions
    def __init__(self):
        self.molecule = self.graph_from_smiles("CC") # Always starts with a single C-C single bond
        self.rdkit_mol = Chem.RWMol(Chem.MolFromSmiles("CC"))
        self.atomic_nums = [6, 8, 7, 16, 9, 17, 15, 35, 53, 5, 14, 16]
        self.atomic_symbols = ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B', 'Si']
        self.bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

        self.node_bank = torch.zeros([11, 11], dtype = torch.float32)
        for i in range(11):
            self.node_bank[i][i] = 1.0

        self.action_space = spaces.Box(low = 0, high = 80, dtype = np.int64, shape = (4,)) # Termination vector, node1, node2, and bond type. Max of 70 nodes and 11 from bank
        self.observation_space = spaces.Graph(node_space=spaces.Box(low=0, high=1, dtype= np.int64, shape=(11,)),
                                              edge_space=spaces.Box(low=0, high=1, dtype = np.int64, shape=(4,)), seed=42) # Bare molecular embeddings

    def reset(self):
        self.molecule = self.graph_from_smiles("CC") # Resets environment to C-C bond
        return self.molecule # Returns the base molecule

    def step(self, action): # Action: [termination binary vector, node1 idx, node2 idx, bond_type from 1-4]

        # Done
        if action[0] == 1 or self.molecule.num_nodes == 70: # An episode is done if the termination vector is 0 or if the molecule size reaches 70 heavy atoms
            done = True

        # Update self.molecule and self.rdkit_mol
        all_atom_attr = torch.vstack((self.molecule.x, self.node_bank)) # Current node embeddings with the node bank concatenated at end

        new_edge_attr = torch.zeros(4, dtype=torch.float32)
        new_edge_attr[action[3] - 1] = 1. # Action specifying bond type is a number from 1 to 4
        new_bond = self.bond_types[int(np.where(new_edge_attr == 1)[0])]

        # If a new atom is being added on to the original molecule
        if action[2] >= self.molecule.num_nodes:
            new_node_attr = all_atom_attr[action[2], :] # Get embedding for new atom

            self.molecule = self.add_node(self.molecule, new_node_attr)
            self.molecule = self.add_edge(self.molecule, action[1], self.molecule.num_nodes - 1, new_edge_attr)

            new_atom = Chem.Atom(self.atomic_nums[int(np.where(new_node_attr.numpy() == 1)[0])])
            self.rdkit_mol.AddAtom(new_atom)
            self.rdkit_mol.AddBond(action[1], self.molecule.num_nodes - 1, new_bond)

        # If only a new bond is being created between two pre-existing atoms
        else:
            self.molecule = self.add_edge(self.molecule, action[1], action[2], new_edge_attr)

            self.rdkit_mol.AddBond(action[1], action[2], new_bond)

        # Update self.rdkit_mol

        # Reward
        reward = QED.weights_mean(self.rdkit_mol.GetMol())

        # Info
        info = self.molecule.num_nodes

        return self.molecule, reward, done, info

    def render(self):
        pass

env = MolBuilderEnv()
print(env.step([0, 0, 3, 1]))