import torch_geometric
import torch
import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from graph_embedding import bare_graph_from_smiles
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def bfs(graph, node):
    """
    Breadth Depth Search
    https://favtutor.com/blogs/breadth-first-search-python
    """
    visited = []
    queue = []

    visited.append(node)
    queue.append(node)

    while queue:  # Creating loop to visit each node
        m = queue.pop(0)

        for neighbour in graph[m]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
    return visited

def order(g):
    """
    Returns the BFS node visiting order from a graph
    """

    adj = g.edge_index.numpy()
    adj = np.swapaxes(adj, 0, 1)

    node = []
    connections = []
    current_connections = []

    for i in range(len(adj)):
        if str(adj[i, 0]) in node:
            current_connections += [str(adj[i, 1])]
        elif str(adj[i, 0]) not in node:
            connections.append(current_connections)  # Incorrectly appends an empty list for the first node
            current_connections = []

            node += [str(adj[i, 0])]
            current_connections += [str(adj[i, 1])]

    connections.append(current_connections)
    connections = connections[1:]  # Correcting for error

    adj_dict = dict()
    for i in range(len(node)):
        adj_dict[node[i]] = connections[i]

    sequence = [int(i) for i in bfs(adj_dict, '0')]

    return sequence, adj_dict

def draw_pyg_graph(g):
    """Draws a PyG graph"""
    nx_g = torch_geometric.utils.to_networkx(g)
    nx.draw(G=nx_g, with_labels=True)
    plt.show()

def add_node(old_g, new_node_embedding, old_node_idx_to_connect_with, new_edge_embedding):
    # New node embeddings x
    if len(old_g.x.shape) != 2: # If the graph only has one node
        new_node_idx = 1
        new_x = np.expand_dims(np.array(old_g.x.tolist()), 0)
        new_x = new_x.tolist()
    else:
        new_node_idx = len(old_g.x)
        new_x = old_g.x.tolist()
    new_x.append(new_node_embedding)

    # New edge index
    if len(old_g.x.shape) != 2: # If the graph only has one node
        new_edge_index = [[new_node_idx, old_node_idx_to_connect_with]]
        new_edge_index.append([old_node_idx_to_connect_with, new_node_idx])
    else:
        new_edge_index = old_g.edge_index.tolist()
        new_edge_index[0] += [new_node_idx, old_node_idx_to_connect_with]
        new_edge_index[1] += [old_node_idx_to_connect_with, new_node_idx]

    # New edge attr
    if len(old_g.x.shape) != 2:
        new_edge_attr = [new_edge_embedding]
    else:
        new_edge_attr = old_g.edge_attr.tolist()
        new_edge_attr.append(new_edge_embedding)
        new_edge_attr.append(new_edge_embedding)

    # Convert to tensors
    new_x = torch.tensor(new_x)
    new_edge_index = torch.LongTensor(new_edge_index)
    new_edge_attr = torch.tensor(new_edge_attr)

    new_g = Data(x = new_x, edge_attr = new_edge_attr, edge_index = new_edge_index)

    return new_g

def add_edge(old_g, new_edge_embedding, node1, node2):
    new_edge_index = old_g.edge_index.tolist()
    new_edge_index[0] += [node1, node2]
    new_edge_index[1] += [node2, node1]
    x = new_edge_index[0]
    y = new_edge_index[1]

    # Reordering new_edge_index to match default organization
    y = [pair for _, pair in sorted(zip(y, x))]
    x = sorted(new_edge_index[0])
    new_edge_index[0] = x
    new_edge_index[1] = y
    new_edge_index = torch.LongTensor(new_edge_index)

    new_edge_attr = old_g.edge_attr.tolist()
    new_edge_attr.append(new_edge_embedding)
    new_edge_attr.append(new_edge_embedding)
    new_edge_attr = torch.tensor(new_edge_attr)

    new_g = Data(x = old_g.x, edge_attr = new_edge_attr, edge_index = new_edge_index)

    return new_g

"""Generating intermediate graphs based on final graph"""
final_g = bare_graph_from_smiles(["c1ccccc1"])[0]
final_g = add_edge(final_g, [0, 0, 0, 1], 1, 3)
ordering, adj_dict = order(final_g)

intermediate_graph = []
num_edges_for_most_recent_node = 0 # Number of edges
current_nodes_in_order_added_to_intermediate = []
dec_ordering = ordering
final_edge_index = np.swapaxes(final_g.edge_index.tolist(), 0, 1).tolist()

for i in range(len(ordering)):
    if i == 0:
        intermediate_graph = Data(x = final_g.x[dec_ordering[i]])
        current_nodes_in_order_added_to_intermediate += [dec_ordering[0]]
        dec_ordering = dec_ordering[1:]
    if i > 0:
        if num_edges_for_most_recent_node == 0: # Procedure for adding a new node

            # Checking how many edges the new node should have with pre-existing nodes
            all_edges_for_current_nodes_in_final_g = []
            for idx in current_nodes_in_order_added_to_intermediate:
                all_edges_for_current_nodes_in_final_g += adj_dict[str(idx)]
            num_edges_for_most_recent_node = all_edges_for_current_nodes_in_final_g.count((str(dec_ordering[i])))

            # Adding a new node to the graph
            old_g = intermediate_graph
            new_node_embedding = final_g.x[dec_ordering[0]].tolist()
            old_node_idx = current_nodes_in_order_added_to_intermediate[-1]
            new_edge_attr_idx = final_edge_index.index([dec_ordering[0], old_node_idx]) # Finding the new_edge_attr
            new_edge_attr = final_g.edge_attr[new_edge_attr_idx].tolist()
            intermediate_graph = add_node(old_g, new_node_embedding, old_node_idx, new_edge_attr)

        if num_edges_for_most_recent_node >= 1:
            old_g = intermediate_graph
            new_edge_embedding = current_nodes_in_order_added_to_intermediate

        print(num_edges_for_most_recent_node)