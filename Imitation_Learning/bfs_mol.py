import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import RWMol
import warnings

import copy
import matplotlib.pyplot as plt

"""
Breaks a single molecule down based on breadth first search into a rollout.
"""

warnings.filterwarnings("ignore", category=DeprecationWarning)

def BFS(graph, node):
    visited = []  # List to keep track of visited nodes.
    queue = []  # Initialize a queue
    visited.append(node)
    queue.append(node)
    order = []

    while queue:
        s = queue.pop(0)
        order.append(s)
        # print (s, end = " ")

        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
    return order

def draw_mol(mol):
    """Displays an image of an RDKit molecule"""
    mol_copy = copy.copy(mol)
    for atom in mol_copy.GetAtoms():
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
    img = Draw.MolToImage(mol_copy)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# def draw_mol_i(mol, i):
#     """Displays an image of an RDKit molecule"""
#     mol_copy = copy.copy(mol)
#     for atom in mol_copy.GetAtoms():
#         atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
#     img = Draw.MolToImage(mol_copy)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.savefig(f'{i}')

def rollout_mol(mol):
    """
    Given a molecule (or SMILES string), splits it based on BFS and outputs terminate, atom1, atom2, bond, and states, the usual observations
    that would occur during an episode of rollout.
    """

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    # Represent a molecular graph as a dictionary
    rdkit.Chem.rdmolops.Kekulize(mol)
    mol_dict = {}
    for atom in mol.GetAtoms():
        atom_idx = str(atom.GetIdx())
        neighbors = [str(neighbor.GetIdx()) for neighbor in atom.GetNeighbors()]
        mol_dict[atom_idx] = neighbors

    # draw_mol(mol)

    # Start the BFS at carbon atom with the most bonds
    max_bonds = 0
    atom_with_max_bonds = None

    has_carbons = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            has_carbons = True
    if has_carbons:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C': # If atom is a carbon
                num_bonds = atom.GetDegree()
                if num_bonds > max_bonds:
                    max_bonds = num_bonds
                    atom_with_max_bonds = atom
    else:
        for atom in mol.GetAtoms():
            num_bonds = atom.GetDegree()
            if num_bonds > max_bonds:
                max_bonds = num_bonds
                atom_with_max_bonds = atom

    # Pass dictionary into BFS method
    order = BFS(mol_dict, str(atom_with_max_bonds.GetIdx()))

    # Rollout
    # Record terminate, atom1, atom2, bond, and the state for each "step"

    # Initial state is one carbon
    states = []
    state = RWMol()
    state.AddAtom(Chem.Atom(atom_with_max_bonds.GetSymbol()))
    states.append(copy.copy(state))

    terminate = []
    atom1 = [] # Lower index of addition
    atom2 = [] # Higher index of addition
    bond = [] # Bond index

    # print(f'Order: {order}')
    # print(f'Mol Dict: {mol_dict}')
    # print()

    mol_atoms = [atom for atom in mol.GetAtoms()]
    atom_bank_symbols = ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B']

    for i in range(len(order) - 1):
        # print(f'{i + 1}')
        state.AddAtom(Chem.Atom(mol_atoms[int(order[i + 1])].GetSymbol())) # Add a new atom

        # Find the indices of other atoms in the state that the new atom should form bonds with
        neighbors = []
        for idx in order[:i + 1]:
            n = mol_dict[idx]
            if order[i + 1] in n:
                neighbors.append(idx)
        # print(f'Added: {order[:i + 1]}')
        # print(f'New atom: {order[i + 1]}')
        # print(f'Connections between new and added: {neighbors}')
        # print()

        # Form a new bond
        for idx in neighbors:
            state.AddBond(order[:i + 1].index(idx), i + 1, order = mol.GetBondBetweenAtoms(int(idx), int(order[i + 1])).GetBondType())
            state.UpdatePropertyCache()
            states.append(copy.copy(state))

            terminate += [0]
            atom1.append(order[:i + 1].index(idx))
            atom2.append(i + 1)
            bond.append(int(mol.GetBondBetweenAtoms(int(idx), int(order[i + 1])).GetBondType()) - 1)
        # draw_mol(state, i)

    terminate[-1] = 1

    return terminate, atom1, atom2, bond, states

if __name__ == '__main__':
    # mol = Chem.MolFromSmiles('Cn1cnc2n(C)c(=O)n(C)c(=O)c12') # Caffeine molecule
    # terminate, atom1, atom2, bond, states = rollout_mol(mol)
    mol = Chem.MolFromSmiles('CCOc1cc2ncc(C#N)c(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C')
    print(len(mol.GetAtoms()))
    terminate, atom1, atom2, bond, states = rollout_mol(mol)
    print(len(states[-1].GetAtoms()))
    draw_mol(states[-1])
    print(Chem.MolToSmiles(states[-1]))
    print(bond.index(3))
    draw_mol(states[23])
    print(terminate[23])
    print(atom1[23])
    print(atom2[23])
    print(bond[23])
    draw_mol(states[24])