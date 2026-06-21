import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, GINEConv, global_mean_pool


class GPSClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        heads: int,
        dropout: float,
        edge_dim: int = 2,
        num_classes: int = 4,
    ):
        super().__init__()
        assert hidden_channels % heads == 0, f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            gine = GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ),
                edge_dim=edge_dim,
            )
            self.convs.append(
                GPSConv(hidden_channels, gine, heads=heads, dropout=dropout, attn_type="multihead")
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, x, edge_index, batch, edge_attr=None, **kwargs):
        x = self.input_proj(x)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        x = global_mean_pool(x, batch)
        return self.head(x)
