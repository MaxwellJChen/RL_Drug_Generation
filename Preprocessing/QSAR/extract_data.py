import pandas as pd


# Ambit_InchiKey, Original_Entry_ID, Entrez_ID, Activity_Flag, pXC50,DB, Original_Assay_ID, Tax_ID, Gene_Symbol, Ortholog_Group, SMILES
with open("/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/Data/DRD2.rtf") as infile:
    drd2 = pd.DataFrame(columns = ["Ambit_InchiKey", "Original_Entry_ID", "Entrez_ID", "Activity_Flag", "pXC50", "DB",
                                   "Original_Assay_ID", "Tax_ID", "Gene_Symbol", "Ortholog_Group", "SMILES"])
    for idx in enumerate(list([line for line in infile]), 0):
        if idx[0] > 7 and idx[0] < 20:
            data = idx[1].split(sep = ',')
            data[-1] = data[-1][:len(data[-1]) - 2]
            drd2.loc[len(drd2.index)] = data
            # print(data)

    drd2.to_csv("test.csv", index = False)