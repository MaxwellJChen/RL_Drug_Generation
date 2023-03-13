import pandas as pd
import numpy as np
import pickle

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw

# SAS
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import matplotlib.pyplot as plt

"""
Data analysis for:
Element type
Bond type
Formal charge
In a ring
Number of hydrogen neighbors
Number of heavy atom neighbors

Number of heavy atoms in molecule
Synthetic accessibility
Drug-likeness
"""

C = pd.read_csv('/Data/combined.csv')

smiles = C.loc[0][0]
mol = Chem.MolFromSmiles(smiles)

atoms = mol.GetAtoms()
for atom in atoms:
    print(atom.GetSymbol()) # Element type
    print(atom.GetFormalCharge()) # Formal charge
    print(atom.IsInRing()) # In a ring
    print(atom.GetTotalNumHs()) # Num of H neighbors
    print(len(atom.GetNeighbors())) # Num of heavy atom neighbors

bonds = mol.GetBonds()
for bond in bonds:
    print(bond.GetBondType()) # Bond type

print(mol.GetNumHeavyAtoms()) # Num of heavy atoms in molecule
SAS = sascorer.calculateScore(mol) # SAS
print(SAS)
print(Chem.QED.default(mol)) # QED