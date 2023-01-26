import numpy as np
import rdkit
from rdkit.Chem import Draw
import pandas as pd
import matplotlib.pyplot as plt

np.random.shuffle(combined)
mols=[rdkit.Chem.MolFromSmiles(x) for x in list(combined[:20, 1])]
print(mols)