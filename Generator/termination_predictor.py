import torch
from torch_geometric.nn import global_mean_pool, GCNConv
from torch.nn import Linear
import torch.nn.functional as F

"""
References
https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing
"""

class termination_predictor():
    # 27 node features
    # 4 edge features

    def __init__(self):
        super(termination_predictor, self).__init__()
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

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fcn1(x)

        return x

