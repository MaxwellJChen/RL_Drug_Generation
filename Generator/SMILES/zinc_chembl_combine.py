import pandas as pd
import numpy as np

"""Combining ZINC and ChEMBL"""
# Loading ZINC and dropping internal duplicates
# zinc = pd.read_csv('//Users/maxwellchen/PycharmProjects/RL_Drug_Generation/Generator/SMILES/zinc_in_vitro.csv', usecols = ['smiles']).drop_duplicates(ignore_index=True)
# print(zinc.info)

# Loading ChEMBL and dropping internal duplicates
# chembl = pd.read_table('/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/Generator/SMILES/chembl_31.txt', usecols = ['canonical_smiles']).drop_duplicates(ignore_index=True)
# chembl.rename(inplace=True, columns={'canonical_smiles': 'smiles'})
# print(chembl.info)

# Combining ChEMBL and ZINC dataframes
# combined = pd.concat([zinc, chembl], ignore_index = True).drop_duplicates(ignore_index=True) # Removing duplicates
# print(combined.info)

# Saving combined datatset
# combined.to_csv('chembl_zinc.csv', index = False)


"""Removing active SMILES for DRD2 or A2AR from dataset"""
# combined = pd.read_csv('/Users/maxwellchen/PycharmProjects/RL_Drug_Generation'
#                        '/Generator/SMILES/chembl_zinc.csv').to_numpy().flatten().tolist() # Converting to a list for easier manipulation

# DRD2
DRD2_sorted = np.load("/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/QSAR/DRD2/SMILES/DRD2_sorted.npy", allow_pickle=True).tolist() # Sorted array based on activity
DRD2_actives = DRD2_sorted[:4613] # 4612 total actives for DRD2

# A2AR
A2AR = pd.read_csv("/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/QSAR/A2AR/SMILES/A2AR.csv", sep = ';')