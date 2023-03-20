import torch_geometric
from torch_geometric.nn import GATv2Conv, global_mean_pool, GCNConv
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch

"""
Architecture
https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=mHSP6-RBOqCE
"""

class GCN_qsar(nn.Module):
    # 27 node features
    # 4 edge features

    def __init__(self):
        super(GCN_qsar, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(27, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.lin = Linear(64, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        x = x.sigmoid()

        return x

    # def __init__(self):
    #     super(GCN_qsar, self).__init__()
    #     self.GCN1 = GCNConv(27, 100)
    #     self.GCN2 = GCNConv(100, 60)
    #     self.GCN3 = GCNConv(60, 30)
    #     self.fcn1 = nn.Linear(30, 20)
    #     self.fcn2 = nn.Linear(20, 1)
    #
    # def forward(self, x, edge_index, batch):
    #     x = self.GCN1(x, edge_index)
    #     x = x.relu()
    #
    #     x = self.GCN2(x, edge_index)
    #     x = x.relu()
    #
    #     x = self.GCN3(x, edge_index)
    #
    #     x = global_mean_pool(x, batch)
    #
    #     x = F.dropout(x, p = 0.5, training = self.training)
    #
    #     x = self.fcn1(x)
    #     x = x.relu()
    #
    #     x = F.dropout(x, p = 0.5, training = self.training)
    #     x = self.fcn2(x)
    #     x = x.sigmoid()
    #
    #     return x

"""
Training
"""

from graph_embedding import graph_from_smiles
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import pickle

# drd2 = pd.read_csv("DRD2.csv").to_numpy()
# drd2 = np.sort(drd2, axis = 0)
# drd2 = drd2[:9226]
#
# train = np.concatenate((drd2[:3690], drd2[4613:8303]), axis = 0)
# test = np.concatenate((drd2[3690:4613], drd2[8303:]), axis = 0)
# train = graph_from_smiles(train[:, 1], [int(y) for y in train[:, 0] == "A"])
# test = graph_from_smiles(test[:, 1], [int(y) for y in test[:, 0] == "A"])
# print("Embedding done!")
#
# p_train = open('DRD2_train_loader', 'wb')
# p_test = open('DRD2_test_loader', 'wb')
# train_loader = DataLoader(train, batch_size = 64, shuffle = True)
# test_loader = DataLoader(test, batch_size = 64, shuffle = True)
# pickle.dump(train_loader, p_train)
# pickle.dump(test_loader, p_test)

model = GCN_qsar()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
criterion = nn.BCELoss()

def train():
    model.train()
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        data = data.y.unsqueeze(1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def evaluate(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# for epoch in range(1, 31):
#     train()
#     train_acc = evaluate(train_loader)
#     test_acc = evaluate(test_loader)
#     print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

out2 = 0
for count, data in enumerate(train_loader):
    if count < 3:
        out = model(data.x, data.edge_index, data.batch)
        if out == out2:
            print("Fail")
        y = data.y.unsqueeze(1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        out2 = out