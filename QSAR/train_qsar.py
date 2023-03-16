from qsar_predictor import QSAR
import torch
from graph_embedding import graph_from_labels

model = QSAR()
g = graph_from_labels(["FC1=CC=C(C(=O)NC2=CC=C(C3=NN(N=N3)CC(=O)N4CCN(CC4)C(=O)C=5OC=CC5)C=C2)C=C1"])[0]

print(model.forward(g, g.edge_index, batch = 1))