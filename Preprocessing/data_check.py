import pandas as pd
import numpy
import pickle

chembl = pd.read_csv("/Data/sample_tsv.txt", sep='\t', lineterminator='\r')
print(chembl)