import pandas as pd
import numpy
import pickle

chembl = pd.read_csv("/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/chembl_31_chemreps.txt", sep='\t', lineterminator='\r')