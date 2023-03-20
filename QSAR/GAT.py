import torch_geometric
from torch_geometric.nn import GATv2Conv, global_mean_pool, GCNConv
import torch.nn as nn
import torch.nn.functional as F

class GAT_qsar(nn.Module):
    # 27 node features
    # 4 edge features

    def __init__(self):
        super(GAT_qsar, self).__init__()
        self.GAT1 = GATv2Conv(27, 100)
        self.GAT2 = GATv2Conv(100, 60)
        self.GAT3 = GATv2Conv(60, 30)
        self.fcn1 = nn.Linear(1, 20)
        self.fcn2 = nn.Linear(20, 1)

    def forward(self, x, edge_index, batch):
        x = self.GAT1(x, edge_index)
        x = x.relu()

        x = self.GAT2(x, edge_index)
        x = x.relu()

        x = self.GAT3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.fcn1(x)
        x = x.relu()

        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.fcn2(x)

        return x

# training
from graph_embedding import graph_from_smiles
from torch_geometric.data import DataLoader



model = GAT_qsar()