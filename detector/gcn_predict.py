import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_max_pool

class GNNFakeNews(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, out_channels):
        super().__init__()
        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x
