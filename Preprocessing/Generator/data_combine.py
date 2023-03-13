import pandas as pd
import numpy as np

# zinc = pd.read_csv('/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/Data/zinc_in_vitro.csv', usecols = ['smiles']).drop_duplicates(ignore_index=True)
# print(zinc.info)
# chembl = pd.read_table('/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/Data/chembl_31.txt', usecols = ['canonical_smiles']).drop_duplicates(ignore_index=True)
# chembl.rename(inplace=True, columns={'canonical_smiles': 'smiles'})
# print(chembl.info)
# combined = pd.concat([zinc, chembl], ignore_index = True).drop_duplicates(ignore_index=True)
# print(combined.info)

# combined.to_pickle("combined.pkl")
# combined.to_csv('combined.csv')

# combined = pd.read_pickle('/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/Data/combined.pkl')
combined = pd.read_csv('/Data/combined.csv')
# print(combined.info)