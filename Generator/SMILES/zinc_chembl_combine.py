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
combined = pd.read_csv('/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/Generator/SMILES/chembl_zinc.csv').to_numpy().flatten() # Converting to a list for easier manipulation

# DRD2
DRD2_sorted = np.load("/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/QSAR/DRD2/SMILES/DRD2_sorted.npy", allow_pickle=True) # Sorted array based on activity
DRD2_actives = DRD2_sorted[:4613, 1].tolist() # 4613 total actives for DRD2
# print(len(DRD2_actives))

# A2AR
A2AR_pchembl_actives = pd.read_csv("/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/QSAR/A2AR/SMILES/A2AR_pChEMBL_avg_actives") # 3077 actives for A2AR
A2AR_pchembl_actives_smiles = A2AR_pchembl_actives['Smiles'].tolist()

# Combining A2AR and DRD2 actives lists
actives = DRD2_actives + A2AR_pchembl_actives_smiles
# print(len(actives))
actives = np.unique(actives).tolist() # 7690 actives in combined list

# Removal
to_remove = []
for i in range(len(combined)):
    print(i)
    if combined[i] in actives:
        to_remove += [i] # Recording indices where combined dataset smiles is found in actives list

# Saving
d = {'smiles': np.delete(combined, to_remove).tolist()} # Creating dictionary to format into Pandas csv
combined_without_actives = pd.DataFrame(d)
print(combined_without_actives.info)
combined_without_actives.to_csv("zinc_chembl_inactives_with_pchembl_A2AR.csv", index = False)