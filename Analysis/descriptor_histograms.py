import pandas as pd

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

# SAS
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# QED
import rdkit.Chem.QED as QED

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

Metrics
QED
SAS
(After QSAR training)
LogP
Affinity for A2AR
Affinity for DRD2
"""

C = pd.read_csv('/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/Generator/SMILES/combined.csv')

smiles = C.loc[0][0]
mol = Chem.MolFromSmiles(smiles)

# Embedded features
# 1. Atom
atoms = mol.GetAtoms()
total_num_hs = 0
for atom in atoms:
    print(atom.GetSymbol()) # Element type
    print(atom.GetNumHeavyNeighbors()) # Num of heavy atom neighbors
    print(atom.GetTotalNumHs()) # Num of H neighbors
    print(atom.GetFormalCharge()) # Formal charge
    print(atom.IsInRing()) # In a ring
    print(atom.GetIsAromatic()) # If in aromatic system
    total_num_hs += atom.GetTotalNumHs()

# 2. Bond
bonds = mol.GetBonds()
for bond in bonds:
    print(bond.GetBondType()) # Bond type

# 3. Molecular
print(mol.GetNumHeavyAtoms()) # Num of heavy atoms in molecule
print(total_num_hs + mol.GetNumHeavyAtoms()) # Total number of atoms in molecule (including hydrogens)

# Metrics
print(sascorer.calculateScore(mol)) # SAS
print(QED.weights_mean(mol)) # QED