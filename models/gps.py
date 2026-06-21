import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, GINEConv, AttentionalAggregation, global_mean_pool, global_max_pool


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
                GPSConv(hidden_channels, gine, heads=heads, dropout=dropout, attn_type="multihead", norm="layer_norm")
            )

        # 4-way readout: root node + attentional aggregation + mean + max
        self.attn_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_channels, 1))
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x, edge_index, batch, edge_attr=None, ptr=None, **kwargs):
        assert ptr is not None, "GPSClassifier.forward requires ptr (batch.ptr) for root-node readout"
        x = self.input_proj(x)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

        root_emb = x[ptr[:-1]]                      # [B, H] — node 0 of each graph is ROOT
        attn_emb = self.attn_pool(x, batch)          # [B, H]
        mean_emb = global_mean_pool(x, batch)        # [B, H]
        max_emb  = global_max_pool(x, batch)         # [B, H]
        g = torch.cat([root_emb, attn_emb, mean_emb, max_emb], dim=-1)  # [B, 4H]
        return self.head(g)
