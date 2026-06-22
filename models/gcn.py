import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        num_classes: int = 4,
        use_text: bool = False,
        text_dim: int = 384,
    ):
        super().__init__()
        self.use_text = use_text
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = nn.Dropout(dropout)

        if use_text:
            self.text_proj = nn.Linear(text_dim, hidden_channels)
            self.text_norm = nn.LayerNorm(hidden_channels)
            self.text_dropout = nn.Dropout(dropout)
            head_in = hidden_channels * 2
        else:
            head_in = hidden_channels
        self.head = nn.Linear(head_in, num_classes)

    def forward(self, x, edge_index, batch, root_text=None, **kwargs):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)
        g = global_mean_pool(x, batch)
        if self.use_text:
            t = self.text_dropout(self.text_norm(self.text_proj(root_text)))
            g = torch.cat([g, t], dim=-1)
        return self.head(g)
