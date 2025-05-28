import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.nn import global_mean_pool


class Net(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_features: int = 64,
        num_layers: int = 3,
        layer_type: str = "GCN",
        heads: int = 4,  # For GAT/Transformer
        dropout: float = 0.5,
    ):
        super().__init__()

        self.layer_type = layer_type
        self.num_layers = num_layers
        self.dropout = dropout

        # Input 
        self.convs = nn.ModuleList()
        self.convs.append(self._make_layer(input_features, hidden_features))

        # Hidden 
        for _ in range(num_layers - 2):
            self.convs.append(self._make_layer(hidden_features, hidden_features))

        # Output 
        self.convs.append(self._make_layer(hidden_features, output_features))

      

    def _make_layer(self, in_dim, out_dim):
        if self.layer_type == "GCN":
            return GCNConv(in_dim, out_dim)
        elif self.layer_type == "GAT":
            return GATConv(in_dim, out_dim, heads=1)
        elif self.layer_type == "graphtransformer":
            return TransformerConv(in_dim, out_dim, heads=1)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

    def forward(self, x, edge_index, batch=None):

        # Message passing layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            #x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer (no activation/dropout)
        x = self.convs[-1](x, edge_index)

        return x
    
    
    #, F.log_softmax(x, dim=1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.skip.reset_parameters()
