from torch_geometric.loader import DataLoader
import pickle
from torch_geometric.nn import GATv2Conv, global_mean_pool, GCNConv
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
import random
from tqdm import tqdm
random.seed(42)

"""
Architecture
References:
https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=mHSP6-RBOqCE
"""

class GCN_qsar(nn.Module):
    # 27 node features
    # 4 edge features

    def __init__(self):
        super(GCN_qsar, self).__init__()
        torch.manual_seed(42)

        # GCN embedding layers
        self.conv1 = GCNConv(27, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)

        # FCN layers
        self.fcn1 = Linear(64, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = x.relu()

        x = self.conv3(x, edge_index)
        x = x.relu()

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p = 0.5, training=self.training)
        x = self.fcn1(x)

        return x

"""MUTAG dataset"""
# from torch_geometric.datasets import TUDataset
#
# dataset = TUDataset(root='data/TUDataset', name='MUTAG')
# torch.manual_seed(12345)
# dataset = dataset.shuffle()
# train_dataset = dataset[:150]
# test_dataset = dataset[150:]
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

"""DRD2"""
drd2 = pickle.load(open("/QSAR/DRD2/Graphs/DRD2_graphs", 'rb'))

# Stratified split
actives = drd2[:4612]
random.shuffle(actives)
inactives = drd2[4612:]
random.shuffle(inactives)

train = actives[:3690] + inactives[:274422]
test = actives[3690:] + inactives[274422:]

"""DRD2 10000"""
# drd2 = pickle.load(open("/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/QSAR/DRD2/Graphs/DRD2_graphs_10000", 'rb'))
# random.shuffle(drd2)
# train = drd2[:8000]
# test = drd2[:2000]

# Test and train dataloaders
train_loader = DataLoader(train, batch_size = 64, shuffle = True)
test_loader = DataLoader(test, batch_size = 64, shuffle = True)

print("Data loaded.")

model = GCN_qsar()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()

"""Training"""
def train(loader):
    model.train()
    tqdm_loader = tqdm(loader, unit = "batch")
    for data in tqdm_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def acc(loader):
    model.eval()

    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

for epoch in range(1, 15):
    train(train_loader)
    train_acc = acc(train_loader)
    test_acc = acc(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# torch.save(model, "LogP_GCN")

# model = torch.load("/Users/maxwellchen/PycharmProjects/RL_Drug_Generation/QSAR/DRD2_GCN")
# print(acc(test_loader))