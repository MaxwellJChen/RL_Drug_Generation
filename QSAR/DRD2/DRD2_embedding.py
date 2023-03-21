import pandas as pd
import numpy as np
import rdkit
from graph_embedding import graph_from_smiles
import pickle

"""Reading DRD2.rtf file"""
# Ambit_InchiKey, Original_Entry_ID, Entrez_ID, Activity_Flag, pXC50,DB, Original_Assay_ID, Tax_ID, Gene_Symbol, Ortholog_Group, SMILES
# with open("/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/QSAR/DRD2.rtf") as infile:
#     drd2 = pd.DataFrame(columns = ["Activity Flag", "SMILES"])
#     for idx in enumerate(list([line for line in infile]), 0):
#         # 7 to 170000
#         # 170001 to 347696
#         if idx[0] >= 170001:
#             data = idx[1].split(sep = ',')
#             data[-1] = data[-1][:len(data[-1]) - 2] #  :)
#             data = [data[3], data[-1]]
#             drd2.loc[len(drd2.index)] = data
#             print(idx[0])
#             # print(data)
#
#     drd2.to_csv("DRD2_2.csv", index = False)

"""Sorting dataset into actives and inactives"""
# drd2 = pd.read_csv("/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/QSAR/DRD2.csv").to_numpy()
# actives = []
# inactives = []
# for i, row in enumerate(drd2):
#     print(i)
#     if row[0] == 'A' and len(actives) == 0:
#          actives = np.expand_dims(row, axis = 0)
#     elif row[0] == 'A':
#          actives = np.concatenate((actives, np.expand_dims(row, axis = 0)), axis = 0)
#     elif row[0] == 'N' and len(inactives) == 0:
#          inactives = np.expand_dims(row, axis = 0)
#     else:
#          inactives = np.concatenate((inactives, np.expand_dims(row, axis = 0)), axis = 0)
#
# print(len(actives))
# print(len(inactives))
# print(inactives)
#
# drd2 = np.concatenate((actives, inactives), axis = 0)
# np.save('DRD2_sorted.npy', drd2)
# print(len(drd2))

"""Removing bad SMILES which are unparsable by RDKit"""
# drd2 = np.load("/QSAR/DRD2/DRD2_sorted.npy", allow_pickle=True)
# bads = [1329, 5763, 8424, 11083, 18814, 21499, 28029, 29367, 47477, 50892, 54866, 59676, 63785, 66876,
#              76880, 83488, 87901, 90238, 90265, 98471, 118642, 123506, 125779, 127218, 149113, 161571, 165098,
#              177323, 187643, 187889, 208922, 219874, 235536, 240627, 270010, 272361, 272394, 280578, 292297, 300576,
#              303653, 312487, 315588, 329337, 331812, 332494, 340613, 341243, 347067, 347688]
# drd2 = np.delete(drd2, bads, axis = 0)
# np.save('DRD2_bads_removed.npy', drd2)
# for i, smiles in enumerate(drd2[:, 1]):
#     print(f'{i} {rdkit.Chem.MolFromSmiles(smiles).GetNumAtoms()}')

"""Embedding sorted dataset into graphs"""
# drd2 = np.load("/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/QSAR/DRD2/DRD2_sorted.npy", allow_pickle = True)
# print(len(drd2))
# drd2 = drd2[:9226]
# train = np.concatenate((drd2[:3690], drd2[4613:8303]), axis = 0)
# test = np.concatenate((drd2[3690:4613], drd2[8303:]), axis = 0)
# train = graph_from_smiles(train[:, 1], [int(y) for y in train[:, 0] == "A"])
# test = graph_from_smiles(test[:, 1], [int(y) for y in test[:, 0] == "A"])
#
# p_train = open('DRD2_train_loader', 'wb')
# p_test = open('DRD2_test_loader', 'wb')
# train_loader = DataLoader(train, batch_size = 64, shuffle = True)
# test_loader = DataLoader(test, batch_size = 64, shuffle = True)
# pickle.dump(train_loader, p_train)
# pickle.dump(test_loader, p_test)
# drd2 = np.load("/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/QSAR/DRD2/DRD2_bads_removed.npy", allow_pickle = True)
# drd2 = graph_from_smiles(drd2[:, 1], [int(y) for y in drd2[:, 0] == "A"])
# file = open('DRD2/DRD2_graphs', 'wb')
# pickle.dump(drd2, file)
# With bads removed, 4612 actives