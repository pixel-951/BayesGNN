import os
import sys

import torch
#import torch_sparse as ts
from torch_geometric.nn import Linear,  SAGEConv












class DirSageConv(torch.nn.Module):
    """
    Implementation of a directional GraphSAGE convolutional layer.

    Args:
        input_features (int): The dimensionality of the input features.
        output_features (int): The dimensionality of the output features.
        alpha (float, optional): The balance parameter for combining
        source-to-target and target-to-source information.
            Default is 1.


    Methods:
        forward(x, edge_index): Forward pass of the directional GraphSAGE convolutional layer.
    """

    def __init__(self, input_features, output_features):
        super(DirSageConv, self).__init__()

        self.source_to_target = SAGEConv(
            input_features, output_features, flow="source_to_target", root_weight=False
        )
        self.target_to_source = SAGEConv(
            input_features, output_features, flow="target_to_source", root_weight=False
        )
        self.linear = Linear(input_features, output_features)
        self.alpha = 0.5

    def forward(self, x, edge_index):
        out = (
            self.linear(x)
            + (1 - self.alpha) * self.source_to_target(x, edge_index)
            + self.alpha * self.target_to_source(x, edge_index)
        )

        return out


"""class DirGCNConv(torch.nn.Module):
   
    def __init__(self, input_dim, output_dim):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Linear transformations for source-to-destination and destination-to-source edges
        self.linear_src_to_dst = Linear(input_dim, output_dim)
        self.linear_dst_to_src = Linear(input_dim, output_dim)

        # Hyperparameter for combining source-to-destination and destination-to-source information
        self.alpha = 1.

        # Normalized adjacency matrices for source-to-destination and destination-to-source edges
        self.adjacency_norm, self.adjacency_transposed_norm = None, None

    def directed_norm(self, adjacency_matrix):
       
        in_deg = ts.sum(adjacency_matrix, dim=0)
        in_deg_inv_sqrt = in_deg.pow_(-0.5)
        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

        out_deg = ts.sum(adjacency_matrix, dim=1)
        out_deg_inv_sqrt = out_deg.pow_(-0.5)
        out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

        adjacency_matrix = ts.mul(adjacency_matrix, out_deg_inv_sqrt.view(-1, 1))
        adjacency_matrix = ts.mul(adjacency_matrix, in_deg_inv_sqrt.view(1, -1))

        return adjacency_matrix

    def forward(self, x, edge_index):
       
        if self.adjacency_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            # Create sparse adjacency matrices for source-to-destination and destination-to-source edges
            adjacency_matrix = ts.SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adjacency_norm = self.directed_norm(adjacency_matrix)

            adjacency_matrix_transposed = ts.SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adjacency_transposed_norm = self.directed_norm(adjacency_matrix_transposed)

        # Apply directed graph convolution
        src_to_dst_term = self.linear_src_to_dst(self.adjacency_norm @ x)
        dst_to_src_term = self.linear_dst_to_src(self.adjacency_transposed_norm @ x)
        return self.alpha * src_to_dst_term + (1 - self.alpha) * dst_to_src_term

"""