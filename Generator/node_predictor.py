import torch
from torch_geometric.nn import global_mean_pool, GCNConv
from torch.nn import Linear
import torch.nn.functional as F

class node_predictor():