import pandas as pd


# Ambit_InchiKey, Original_Entry_ID, Entrez_ID, Activity_Flag, pXC50,DB, Original_Assay_ID, Tax_ID, Gene_Symbol, Ortholog_Group, SMILES
with open("/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/QSAR/DRD2.rtf") as infile:
    drd2 = pd.DataFrame(columns = ["Activity Flag", "SMILES"])
    for idx in enumerate(list([line for line in infile]), 0):
        # 7 to 170000
        # 170001 to 347696
        if idx[0] >= 170001:
            data = idx[1].split(sep = ',')
            data[-1] = data[-1][:len(data[-1]) - 2] #  :)
            data = [data[3], data[-1]]
            drd2.loc[len(drd2.index)] = data
            print(idx[0])
            # print(data)

    drd2.to_csv("DRD2_2.csv", index = False)
