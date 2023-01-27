import pandas as pd
import numpy as np
import pickle

import rdkit
from rdkit import Chem

# SAS
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

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

C = pd.read_csv('/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/Data/combined.csv')

# Element type
# Bond type
# Formal charge
# In a ring
# Num of H neighbors
# Num of heavy atom neighbors
# Num of heavy atoms in molecule

# SAS

# QED