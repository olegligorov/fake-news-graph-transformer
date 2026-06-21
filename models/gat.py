import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class GATClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, heads: int, dropout: float, num_classes: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.projs = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        self.projs.append(nn.Linear(hidden_channels * heads, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
            self.projs.append(nn.Linear(hidden_channels * heads, hidden_channels))

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, **kwargs):
        for conv, bn, proj in zip(self.convs, self.bns, self.projs):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = proj(x)
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.head(x)
