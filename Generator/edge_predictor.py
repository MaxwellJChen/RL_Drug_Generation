import torch
from torch_geometric.nn import global_mean_pool, GCNConv
from torch.nn import Linear
import torch.nn.functional as F


class edge_predictor():

    def __init__(self):
        super(edge_predictor, self).__init__()
        torch.manual_seed(42)