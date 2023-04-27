import pandas as pd
import numpy as np

from rdkit import Chem

# SAS
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# QED
import rdkit.Chem.QED as QED

# MW
import rdkit.Chem.Descriptors as Descriptors

"""
Embedded Features
1. Atom
Type of atom
Number of heavy neighbors
Number of hydrogen atom neighbors
Formal charge
Presence in a ring
If in aromatic system

2. Bond
Bond types

3. Molecular
Number of heavy atoms
Total atoms (including hydrogens)
Molecular weight

Metrics
QED
SAS
(After QSAR training)
LogP
Affinity for A2AR
Affinity for DRD2
"""

def update_hist(hist, value):
    """
    Accepts a value of a single descriptor and updates histogram array. Histogram array: (values, counts).
    Works "half" in place.
    """

    if len(hist) == 0: # Histogram is empty
        hist = [[value], [1]]
    elif value in hist[0]: # Value already recorded in histogram
        hist[1][hist[0].index(value)] += 1
    else: # New value for descriptor
        hist = [hist[0] + [value], hist[1] + [1]]

    return hist

C = pd.read_csv('/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/Generator/SMILES/zinc_chembl.csv')
smiles = C['smiles'].tolist() # List of generator training smiles

# Histograms
element_type = []
num_heavy_atom_neighbors = []
num_h_neighbors = []
formal_charge = []
in_a_ring = []
is_aromatic = []

bond_type = []

num_heavy_atoms = []
total_num_atoms = []
molecular_weight = []

sas = []
qed = []

n = 13
# Dataset has total of 2,562,583 molecules, so infeasible to generate full histograms in one go. Breaking up into groups of 200,000.
# 1: 0, 200000
# 2: 200000, 400000
# 3: 400000, 600000
# 4: 600000, 800000
# 5: 800000, 1000000
# 6: 1000000, 1200000
# 7: 1200000, 1400000
# 8: 1400000, 1600000
# 9: 1600000, 1800000
# 10: 1800000, 2000000
# 11: 2000000, 2200000
# 12: 2200000, 2400000
# 13: 2400000, len(smiles)
for i in range(2400000, len(smiles)):
    print(i)
    if i == 2465510: # [11:30:38] Can't kekulize mol.  Unkekulized atoms: 1 2 3 4 5 6 10 11 15 16 17 19 20 21
        continue

    mol = Chem.MolFromSmiles(smiles[i])

    # Embedded features
    # 1. Atom
    atoms = mol.GetAtoms()
    total_num_hs = 0
    for atom in atoms:
        element_type = update_hist(element_type, atom.GetSymbol()) # Element type
        num_heavy_atom_neighbors = update_hist(num_heavy_atom_neighbors, atom.GetDegree()) # Num of heavy atom neighbors
        num_h_neighbors = update_hist(num_h_neighbors, atom.GetTotalNumHs()) # Num of H neighbors
        formal_charge = update_hist(formal_charge, atom.GetFormalCharge()) # Formal charge
        in_a_ring = update_hist(in_a_ring, atom.IsInRing()) # In a ring
        is_aromatic = update_hist(is_aromatic, atom.GetIsAromatic()) # If in aromatic system

        total_num_hs += atom.GetTotalNumHs()

    # 2. Bond
    bonds = mol.GetBonds()
    for bond in bonds:
        bond_type = update_hist(bond_type, str(bond.GetBondType())) # Bond type

    # 3. Molecular
    num_heavy_atoms = update_hist(num_heavy_atoms, mol.GetNumHeavyAtoms()) # Num of heavy atoms in molecule
    total_num_atoms = update_hist(total_num_atoms, total_num_hs + mol.GetNumHeavyAtoms()) # Total number of atoms in molecule (including hydrogens)
    molecular_weight = update_hist(molecular_weight, Descriptors.ExactMolWt(mol))

    # Metrics
    sas = update_hist(sas, sascorer.calculateScore(mol)) # SAS
    qed = update_hist(qed, QED.weights_mean(mol)) # QED

# Convert to numpy arrays
element_type = np.array(element_type, dtype = object)
num_heavy_atom_neighbors = np.array(num_heavy_atom_neighbors)
num_h_neighbors = np.array(num_h_neighbors)
formal_charge = np.array(formal_charge)
in_a_ring = np.array(in_a_ring)
is_aromatic = np.array(is_aromatic)

bond_type = np.array(bond_type, dtype = object)

num_heavy_atoms = np.array(num_heavy_atoms)
total_num_atoms = np.array(total_num_atoms)
molecular_weight = np.array(molecular_weight)

sas = np.array(sas)
qed = np.array(qed)

# Checking histograms
# print(element_type)
# print(num_heavy_atom_neighbors)
# print(num_h_neighbors)
# print(formal_charge)
# print(in_a_ring)
# print(is_aromatic)
#
# print(bond_type)
#
# print(num_heavy_atoms)
# print(total_num_atoms)
# print(molecular_weight)
#
# print(sas)
# print(qed)

# Saving histograms
np.save(f"Histograms/{n}/element_type_{n}", element_type, allow_pickle=True)
np.save(f"Histograms/{n}/num_heavy_atom_neighbors_{n}", num_heavy_atom_neighbors, allow_pickle=True)
np.save(f"Histograms/{n}/num_h_neighbors_{n}", num_h_neighbors, allow_pickle=True)
np.save(f"Histograms/{n}/formal_charge_{n}", formal_charge, allow_pickle=True)
np.save(f"Histograms/{n}/in_a_ring_{n}", in_a_ring, allow_pickle=True)
np.save(f"Histograms/{n}/is_aromatic_{n}", is_aromatic, allow_pickle=True)

np.save(f"Histograms/{n}/bond_type_{n}", bond_type, allow_pickle=True)

np.save(f"Histograms/{n}/num_heavy_atoms_{n}", num_heavy_atoms, allow_pickle=True)
np.save(f"Histograms/{n}/total_num_atoms_{n}", total_num_atoms, allow_pickle=True)
np.save(f"Histograms/{n}/molecular_weight_{n}", molecular_weight, allow_pickle=True)

np.save(f"Histograms/{n}/sas_{n}", sas, allow_pickle=True)
np.save(f"Histograms/{n}/qed_{n}", qed, allow_pickle=True)