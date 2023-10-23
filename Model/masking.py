import torch
from rdkit import Chem
from rdkit.Chem import RWMol
import copy
from Model.graph_embedding import visualize

"""
Helper functions for invalid action masking SURGE probability distributions during SURGE acting, supervised learning, and reinforcement learning.

All validity checks for invalid action masking (separated for each of SURGE's 4 outputs):
    Termination:
        1. Must terminate if all atoms have reached maximum valence
    Nmol:
        2. Cannot choose an atom if it has reached its maximum valence
    Nfull:
        3. Cannot choose an atom if it has reached its maximum valence
        4. Cannot choose an atom if it is the same as nmol
        5. Cannot choose an atom if it is already in a ring with nmol
        6. Cannot choose an atom if it already has a bond with nmol
        7. Cannot choose an atom if forming a bond between nmol and itself would result in a ring size greater than 7
    Bond:
        8. Cannot choose a bond order too great for either nmol or nfull's remaining valence

Note: The aggregate mask function repeats operations less but can only be used when nmol and nfull have both been selected. 
It cannot be applied to the SURGE act function but can be used in supervised learning. The separated masking functions are more flexible,
but repeat operations between them.
"""

max_valences = {'C': 4, 'O': 2, 'N': 3, 'S': 6, 'F': 1, 'Cl': 1, 'P': 5, 'Br': 1, 'I': 1, 'B': 3}
atom_bank = ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B']

def t_mask(state):
    """
    Returns the termination mask for a state.
    Termination:
        1. Must terminate if all atoms have reached maximum valence
    """

    t_mask = torch.ones(2)
    valence_diffs = [max_valences[atom.GetSymbol()] - sum([int(bond.GetBondType()) for bond in atom.GetBonds()]) - atom.GetFormalCharge() for atom in state.GetAtoms()]

    if all([diff == 0 for diff in valence_diffs]): # 1. All atoms are at their mask valence. Must terminate generation.
        t_mask[0] = 0

    return t_mask

def nmol_mask(state):
    """
    Returns the nmol mask for a state.
    Nmol:
        2. Cannot choose an atom if it has reached its maximum valence
    """

    nmol_mask = torch.ones(state.GetNumHeavyAtoms())
    valence_diffs = [max_valences[atom.GetSymbol()] - sum([int(bond.GetBondType()) for bond in atom.GetBonds()]) - atom.GetFormalCharge() for atom in state.GetAtoms()]

    for i in range(len(valence_diffs)): # 2. A specific atom is at its max valence, so it cannot be chosen.
        if valence_diffs[i] == 0:
            nmol_mask[i] = 0

    return nmol_mask

def nfull_mask(state, nmol: int):
    """
    Returns the nfull mask for a state.
    Nfull:
        3. Cannot choose an atom if it has reached its maximum valence
        4. Cannot choose an atom if it is the same as nmol
        5. Cannot choose an atom if it is already in a ring with nmol
        6. Cannot choose an atom if it already has a bond with nmol
        7. Cannot choose an atom if forming a bond between nmol and itself would result in a ring size greater than 7
    """

    nfull_mask = torch.ones(state.GetNumHeavyAtoms() + len(atom_bank))
    valence_diffs = [max_valences[atom.GetSymbol()] - sum([int(bond.GetBondType()) for bond in atom.GetBonds()]) - atom.GetFormalCharge() for atom in state.GetAtoms()]

    for i in range(len(valence_diffs)):  # 3. A specific atom is at its max valence, so it cannot be chosen.
        if valence_diffs[i] == 0:
            nfull_mask[i] = 0

    # 4. Cannot select the same atom in nfull as in nmol.
    nfull_mask[nmol] = 0

    ring_info = state.GetRingInfo()
    for i in range(state.GetNumHeavyAtoms()):
        if i == nmol:
            continue  # No need to check the atom that nmol specifies
        if len(ring_info.AtomRings()) != 0 and ring_info.AreAtomsInSameRing(i, nmol):  # 5. Cannot choose an atom if it is already in a ring with nmol.
            nfull_mask[i] = 0

        elif state.GetBondBetweenAtoms(i, nmol) is not None:  # 6. Cannot choose an atom if it already has a bond with nmol
            nfull_mask[i] = 0

        # 7. Cannot choose an atom if forming a bond between nmol and nfull would result in a ring size greater than 7.
        elif not valence_diffs[i] == 0:  # ith atom can make a bond
            test_state = RWMol(copy.copy(state))
            test_state.AddBond(nmol, i, order=Chem.BondType.SINGLE)  # Form hypothetical single bond between atoms
            Chem.SanitizeMol(test_state, catchErrors=True)  # Ensures that the newly formed ring is detected by RDKit
            test_ring_info = test_state.GetRingInfo()
            if len(test_ring_info.AtomRings()) != 0:
                test_ring_sizes = [len(ring) for ring in test_ring_info.AtomRings()]
                if max(test_ring_sizes) > 7:
                    nfull_mask[i] = 0

    return nfull_mask

def b_mask(state, nmol: int, nfull: int):
    """
    Returns the bond mask for a state.
    Bond:
        8. Cannot choose a bond order too great for either nmol or nfull's remaining valence
    """

    b_mask = torch.ones(3)

    # 8. Cannot choose a bond order too great for either nmol or nfull's remaining valence
    nmol_atom = state.GetAtomWithIdx(nmol)
    nmol_valence = max_valences[nmol_atom.GetSymbol()] - sum([int(bond.GetBondType()) for bond in nmol_atom.GetBonds()]) - nmol_atom.GetFormalCharge()  # Calculating remaining valence of atom
    if nfull >= state.GetNumHeavyAtoms():
        nfull_valence = max_valences[atom_bank[nfull - state.GetNumHeavyAtoms()]]
    else:
        nfull_atom = state.GetAtomWithIdx(nfull)
        nfull_valence = max_valences[nfull_atom.GetSymbol()] - sum(
            [int(bond.GetBondType()) for bond in nfull_atom.GetBonds()]) - nfull_atom.GetFormalCharge()

    for i in range(min(nmol_valence, nfull_valence), 3):
        b_mask[i] = 0

    return b_mask

def mask(state, nmol, nfull):
    """
    Accepts a state (kekulized Mol or RWMol) and returns the masking for the probability distribution of the state. Must include nmol to mask nfull.

    All validity checks for invalid action masking:
        Termination:
            1. Must terminate if all atoms have reached maximum valence
        Nmol:
            2. Cannot choose an atom if it has reached its maximum valence
        Nfull:
            3. Cannot choose an atom if it has reached its maximum valence
            4. Cannot choose an atom if it is the same as nmol
            5. Cannot choose an atom if it is already in a ring with nmol
            6. Cannot choose an atom if it already has a bond with nmol
            7. Cannot choose an atom if forming a bond between nmol and itself would result in a ring size greater than 7
        Bond:
            8. Cannot choose a bond order too great for either nmol or nfull's remaining valence
    """

    # Initialize masks to all ones (every option is available)
    t_mask = torch.ones(2)
    num_atoms = state.GetNumHeavyAtoms()
    nmol_mask = torch.ones(num_atoms)
    nfull_mask = torch.ones(num_atoms + len(atom_bank))
    b_mask = torch.ones(3)

    valence_diffs = [max_valences[atom.GetSymbol()] - sum([int(bond.GetBondType()) for bond in atom.GetBonds()]) -
                     atom.GetFormalCharge() for atom in state.GetAtoms()]  # List of ints representing difference in maximum valence of an atom and its actual valence
    has_max_valence = [diff == 0 for diff in valence_diffs]  # List of booleans representing if an atom is at its maximum valence

    # Termination
    if all(has_max_valence):  # 1. All atoms are at their mask valence. Must terminate generation.
        t_mask[0] = 0  # 1 indicates termination. This makes the choice of 0 impossible.

    # Nmol and Nfull
    for i in range(len(has_max_valence)):  # 2 and 3. A specific atom is at its max valence, so it cannot be chosen.
        if has_max_valence[i]:
            nmol_mask[i] = 0
            nfull_mask[i] = 0

    # Nfull
    # 4. Cannot select the same atom in nfull as in nmol.
    nfull_mask[nmol] = 0

    ring_info = state.GetRingInfo()
    for i in range(num_atoms):
        if i == nmol:
            continue  # No need to check the atom that nmol specifies
        if len(ring_info.AtomRings()) != 0 and ring_info.AreAtomsInSameRing(i, nmol):  # 5. Cannot choose an atom if it is already in a ring with nmol.
            nfull_mask[i] = 0

        elif state.GetBondBetweenAtoms(i, nmol) is not None:  # 6. Cannot choose an atom if it already has a bond with nmol
            nfull_mask[i] = 0

        # 7. Cannot choose an atom if forming a bond between nmol and nfull would result in a ring size greater than 7.
        elif not has_max_valence[i]:  # Atom i can make a bond
            test_state = RWMol(copy.copy(state))
            test_state.AddBond(nmol, i, order=Chem.BondType.SINGLE)  # Form hypothetical single bond between atoms
            Chem.SanitizeMol(test_state, catchErrors=True)  # Ensures that the newly formed ring is detected by RDKit
            test_ring_info = test_state.GetRingInfo()
            if len(test_ring_info.AtomRings()) != 0:
                test_ring_sizes = [len(ring) for ring in test_ring_info.AtomRings()]
                if max(test_ring_sizes) > 7:
                    nfull_mask[i] = 0

    # Bond
    # 8. Cannot choose a bond order too great for either nmol or nfull's remaining valence
    nmol_atom = state.GetAtomWithIdx(nmol)
    nmol_valence = max_valences[nmol_atom.GetSymbol()] - sum([int(bond.GetBondType()) for bond in nmol_atom.GetBonds()]) - nmol_atom.GetFormalCharge() # Calculating remaining valence of atom
    if nfull >= num_atoms:
        nfull_valence = max_valences[atom_bank[nfull - num_atoms]]
    else:
        nfull_atom = state.GetAtomWithIdx(nfull)
        nfull_valence = max_valences[nfull_atom.GetSymbol()] - sum([int(bond.GetBondType()) for bond in nfull_atom.GetBonds()]) - nfull_atom.GetFormalCharge()

    for i in range(min(nmol_valence, nfull_valence), 3):
        b_mask[i] = 0

    return t_mask, nmol_mask, nfull_mask, b_mask

if __name__ == '__main__':
    atom_bank = ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Br', 'I', 'B']
    state = RWMol(Chem.MolFromSmiles('C=CCl'))
    visualize(state)
    print(b_mask(state, 0, 6))