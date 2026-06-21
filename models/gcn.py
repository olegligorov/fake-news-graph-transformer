import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float, num_classes: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, **kwargs):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.head(x)
