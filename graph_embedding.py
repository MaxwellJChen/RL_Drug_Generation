import torch_geometric
import torch
import rdkit
from rdkit import Chem
import numpy as np

"""
https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
https://arxiv.org/pdf/2012.04444.pdf
12 one-hot vector specifying the type of atom
H, C, O, N, S, F, Cl, P, Br, I, B, Si, Sn
6 number of heavy neighbours as one-hot vector
5 number of hydrogen atoms as one-hot vector
1 formal charge
1 is in a ring
1 is in aromatic system
"""

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element of the permitted list
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom):
    """
    Takes an RDKit atom object as input and gives a 1D-numpy array of atom features as output
    """
    # define list of permitted atoms
    permitted_list_of_atoms = ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B', 'Si', 'Sn']

    # compute atom features
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms) # 12 one-hot vector specifying type of atom
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "More"]) # 6 heavy neighbors vector
    n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"]) # 5 Number of hydrogen atoms vector
    formal_charge_enc = [int(int(atom.GetFormalCharge()) != 0)] # 1 formal charge binary encoding
    is_in_a_ring_enc = [int(atom.IsInRing())] # 1 ring inclusion binary encoding
    is_aromatic_enc = [int(atom.GetIsAromatic())] # 1 aromatic binary encoding
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + n_hydrogens_enc + formal_charge_enc + is_in_a_ring_enc + is_aromatic_enc
    return np.array(atom_feature_vector)

def get_bond_features(bond):
    """
    Takes an RDKit bond object as input and gives a 1D-numpy array of bond features as output
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_feature_vector = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    return np.array(bond_feature_vector)

def graph_from_labels(x_smiles):
    """
    Inputs:
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings

    Outputs:
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    """

    data_list = []

    for smiles in x_smiles:

        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)

        X = torch.tensor(X, dtype=torch.float)

        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(Chem.GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))

        for (k, (i, j)) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

        EF = torch.tensor(EF, dtype=torch.float)

        # construct Pytorch Geometric data object and append to data list
        data_list.append(torch_geometric.data.Data(x=X, edge_index=E, edge_attr=EF))

    return data_list

g = graph_from_labels(["FC1=CC=C(C(=O)NC2=CC=C(C3=NN(N=N3)CC(=O)N4CCN(CC4)C(=O)C=5OC=CC5)C=C2)C=C1"])[0]
print(g.num_node_features)
print(g.num_edge_features)