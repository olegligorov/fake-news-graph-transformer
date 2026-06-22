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
        use_text: bool = False,
        text_dim: int = 384,
        readout: str = "4way",
    ):
        super().__init__()
        assert hidden_channels % heads == 0, f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
        assert readout in ("4way", "mean"), f"readout must be '4way' or 'mean', got {readout!r}"
        self.use_text = use_text
        self.readout = readout

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

        if readout == "4way":
            self.attn_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_channels, 1))
            readout_dim = hidden_channels * 4
        else:
            readout_dim = hidden_channels

        if use_text:
            self.text_proj = nn.Linear(text_dim, hidden_channels)
            self.text_norm = nn.LayerNorm(hidden_channels)
            self.text_dropout = nn.Dropout(dropout)
            readout_dim += hidden_channels
        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x, edge_index, batch, edge_attr=None, ptr=None, root_text=None, **kwargs):
        x = self.input_proj(x)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

        if self.readout == "4way":
            assert ptr is not None, "GPSClassifier.forward requires ptr for the 4-way readout"
            root_emb = x[ptr[:-1]]
            attn_emb = self.attn_pool(x, batch)
            mean_emb = global_mean_pool(x, batch)
            max_emb  = global_max_pool(x, batch)
            g = torch.cat([root_emb, attn_emb, mean_emb, max_emb], dim=-1)
        else:
            g = global_mean_pool(x, batch)
        if self.use_text:
            t = self.text_dropout(self.text_norm(self.text_proj(root_text)))
            g = torch.cat([g, t], dim=-1)
        return self.head(g)
