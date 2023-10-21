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

def BFS(graph: dict, node: str):
    """
    Implementation of the breadth-first search given a graph represented as a dictionary.
    Accepts a graph represented as a dictionary and the index of the starting node. Returns the order of the nodes by index.
    """

    visited = []  # List to keep track of visited nodes.
    queue = []  # Initialize a queue
    visited.append(node)
    queue.append(node)
    order = []

    while queue:
        s = queue.pop(0)
        order.append(s)

        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
    return order

def visualize_rollout(mol, i):
    """
    Displays an image of an RDKit molecule. Index number i allows png to be saved as a step during rollout.
    """

    mol_copy = copy.copy(mol)
    for atom in mol_copy.GetAtoms():
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
    img = Draw.MolToImage(mol_copy)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.savefig(f'{i}')

def rollout(mol, visualize = False):
    """
    Given a molecule (or SMILES string), splits it with on BFS and outputs terminate, nmol, nfull, bond, and states,
    i.e., the usual observations that would occur during an episode of rollout.
    visualize argument specifies whether or not to output and save rollout phase.
    """

    """Obtaining the BFS order for rollout."""
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    # Represent a molecular graph as a dictionary
    rdkit.Chem.rdmolops.Kekulize(mol)
    mol_dict = {}
    for atom in mol.GetAtoms():
        atom_idx = str(atom.GetIdx())
        neighbors = [str(neighbor.GetIdx()) for neighbor in atom.GetNeighbors()]
        mol_dict[atom_idx] = neighbors

    # Start the BFS at carbon atom with the most bonds
    max_bonds = 0 # Stores the number of bonds for the atom with the most bonds
    atom_with_max_bonds = None # Records the atom with the most bonds
    has_carbons = False
    for atom in mol.GetAtoms(): # Test if there are even carbons in the molecule
        if atom.GetSymbol() == 'C':
            has_carbons = True
    if has_carbons:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C': # If atom is a carbon
                num_bonds = atom.GetDegree()
                if num_bonds > max_bonds:
                    max_bonds = num_bonds
                    atom_with_max_bonds = atom # Find the carbon with the most bonds
    else:
        for atom in mol.GetAtoms():
            num_bonds = atom.GetDegree()
            if num_bonds > max_bonds:
                max_bonds = num_bonds
                atom_with_max_bonds = atom # Find the atom with the most bonds

    # Pass dictionary into BFS method to retrieve the order of nodes to append
    order = BFS(mol_dict, str(atom_with_max_bonds.GetIdx()))

    """Executing rollout given the BFS order."""
    # Record terminate, nmol, nfull, bond, and the state for each "step"

    # Initial state begins with the atom holding the most bonds (most likely a carbon)
    states = []
    state = RWMol()
    state.AddAtom(Chem.Atom(atom_with_max_bonds.GetSymbol()))
    states.append(copy.copy(state))

    if visualize == True:
        t = 1 # Iterator for visualizing
        visualize_rollout(state, t)

    terminate = []
    nmol = [] # Lower index of addition
    nfull = [] # Higher index of addition
    bond = [] # Bond index

    mol_atoms = [atom for atom in mol.GetAtoms()] # Contains all the atoms in the complete molecule (in a different order from BFS)
    atom_bank = ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B']

    for i in range(len(order) - 1):
        state.AddAtom(Chem.Atom(mol_atoms[int(order[i + 1])].GetSymbol())) # Add a new atom

        # Check if any of the atoms in the intermediate rollout stage (i.e., already in the graph) should form bonds with the new atom
        neighbors = [] # Contains the indices (from mol_atoms) of the atoms already in the graph that should form bonds with the new atom
        for idx in order[:i + 1]:
            n = mol_dict[idx]
            if order[i + 1] in n:
                neighbors.append(idx)

        # Form a new bond
        for j in range(len(neighbors)):
            b = mol.GetBondBetweenAtoms(int(neighbors[j]), int(order[i + 1])).GetBondType() # Get the bond between the two molecules in the original molecule
            state.AddBond(order[:i + 1].index(neighbors[j]), i + 1, order = b) # Add a bond to the molecule

            # Update variables containing entirety of rollout
            state.UpdatePropertyCache()
            states.append(copy.copy(state))

            # Save and visualize currount rollout step
            if visualize == True:
                t += 1
                visualize_rollout(state, t)

            # Update supervised learning data
            terminate += [0]
            nmol.append(order[:i + 1].index(neighbors[j]))

            if j == 0:
                nfull.append(state.GetNumHeavyAtoms() + atom_bank.index(Chem.Atom(mol_atoms[int(order[i + 1])]).GetSymbol()) - 1)
            else:
                nfull.append(i + 1)
            bond.append(int(mol.GetBondBetweenAtoms(int(neighbors[j]), int(order[i + 1])).GetBondType()) - 1)

    terminate += [1]
    nmol += [0]
    nfull += [0]
    bond += [0]

    actions = dict()
    actions['t'] = terminate
    actions['nmol'] = nmol
    actions['nfull'] = nfull
    actions['b'] = bond

    return actions, states

if __name__ == '__main__':
    actions, states = rollout(Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C'))
    print(actions)
    print(states)
    # mol = Chem.MolFromSmiles('c1ccccc1')
    # print(type(mol))