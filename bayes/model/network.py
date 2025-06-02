from enum import Enum

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    ARMAConv,
    ChebConv,
    CuGraphSAGEConv,
    GATConv,
    GCNConv,
    GraphConv,
    LEConv,
    Linear,
    MFConv,
    SAGEConv,
    SGConv,
    SSGConv,
    TransformerConv,
)
from .GNN_layers import DirSageConv
from torch_geometric.nn import global_mean_pool


class LayerType(Enum):
    GCNCONV = "GCNConv"
    SAGECONV = "SAGEConv"
    GATCONV = "GATConv"
    TRANSFORMERCONV = "TransformerConv"
    LINEAR = "Linear"
    CHEBCONV = "ChebConv"
    CUGRAPHSAGECONV = "CuGraphSAGEConv"
    GRAPHCONV = "GraphConv"
    ARMACONV = "ARMAConv"
    SGCONV = "SGConv"
    MFCONV = "MFConv"
    SSGCONV = "SSGConv"
    LECONV = "LEConv"
    DIRSAGECONV = "DirSageConv"
    DIRGCNCONV = "DirGCNConv"

class Net(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_features: int = 64,
        num_layers: int = 3,
        layer_type: str = "GCNConv",
        heads: int = 1, 
        dropout: float = 0.5,
    ):
        super().__init__()

        self.layer_type = layer_type
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(self._get_conv_layer(input_features, hidden_features))

        for _ in range(num_layers - 2):
            self.convs.append(self._get_conv_layer(hidden_features, hidden_features))

        self.convs.append(self._get_conv_layer(hidden_features, output_features))

      

    

    def forward(self, x, edge_index, batch=None):

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            #x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)

        return x
    
    
    #, F.log_softmax(x, dim=1)


    def _get_conv_layer(self, input_features, output_features):
     
        if self.layer_type == LayerType.LINEAR.value:
            return Linear(input_features, output_features)
        elif self.layer_type == LayerType.GCNCONV.value:
            return GCNConv(input_features, output_features)
        elif self.layer_type == LayerType.SAGECONV.value:
            return SAGEConv(input_features, output_features)
        elif self.layer_type == LayerType.GATCONV.value:
            return GATConv(input_features, output_features)
        elif self.layer_type == LayerType.TRANSFORMERCONV.value:
            return TransformerConv(input_features, output_features)
        elif self.layer_type == LayerType.CHEBCONV.value:
            return ChebConv(input_features, output_features, K=3)
        elif self.layer_type == LayerType.CUGRAPHSAGECONV.value:
            return CuGraphSAGEConv(input_features, output_features)
        elif self.layer_type == LayerType.GRAPHCONV.value:
            return GraphConv(input_features, output_features)
        elif self.layer_type == LayerType.ARMACONV.value:
            return ARMAConv(input_features, output_features)
        elif self.layer_type == LayerType.SGCONV.value:
            return SGConv(input_features, output_features)
        elif self.layer_type == LayerType.MFCONV.value:
            return MFConv(input_features, output_features)
        elif self.layer_type == LayerType.SSGCONV.value:
            return SSGConv(input_features, output_features, alpha=0.5)
        elif self.layer_type == LayerType.LECONV.value:
            return LEConv(input_features, output_features)
        elif self.layer_type == LayerType.DIRSAGECONV.value:
            return DirSageConv(input_features, output_features)
        else:
            raise ValueError(f"Unsupported layer_type: {self.layer_type}")

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.skip.reset_parameters()
