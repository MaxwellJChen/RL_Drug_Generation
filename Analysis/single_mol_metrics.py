import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.QED as QED # QED


mol = Chem.MolFromSmiles('O[Cl+3](O)(O)O')
print(QED.weights_mean(mol))