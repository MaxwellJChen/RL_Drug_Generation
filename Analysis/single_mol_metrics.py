import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.QED as QED # QED

import pandas as pd

smiles = pd.read_csv('/Imitation_Learning/in_trials.csv')['smiles'].tolist()
qed = 0.
m = 0.
for smile in smiles:
    try:
        qed += QED.weights_mean(Chem.MolFromSmiles(smile))
        m += 1.
    except Exception as e:
        pass

print(qed)
print(m)
print(qed/m)