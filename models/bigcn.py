import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class BiGCNClassifier(nn.Module):
    """Bidirectional GCN: separate top-down and bottom-up GCN streams.

    Replicates the two-stream structure from Bian et al. (2020) "Rumor Detection
    on Social Media with Bi-Directional Graph Convolutional Networks".
    Top-down edges (parent→child, direction flag=1) model rumor propagation.
    Bottom-up edges (child→parent, direction flag=0) model reply aggregation.
    Both streams share the same hidden dimension; outputs are concatenated before
    the classification head.
    """

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

        self.td_convs = nn.ModuleList()  # top-down stream
        self.bu_convs = nn.ModuleList()  # bottom-up stream
        self.td_bns = nn.ModuleList()
        self.bu_bns = nn.ModuleList()

        self.td_convs.append(GCNConv(in_channels, hidden_channels))
        self.bu_convs.append(GCNConv(in_channels, hidden_channels))
        self.td_bns.append(nn.BatchNorm1d(hidden_channels))
        self.bu_bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.td_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bu_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.td_bns.append(nn.BatchNorm1d(hidden_channels))
            self.bu_bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = nn.Dropout(dropout)

        if use_text:
            self.text_proj = nn.Linear(text_dim, hidden_channels)
            self.text_norm = nn.LayerNorm(hidden_channels)
            self.text_dropout = nn.Dropout(dropout)
            head_in = hidden_channels * 3  # td + bu + text
        else:
            head_in = hidden_channels * 2  # td + bu
        self.head = nn.Linear(head_in, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None, root_text=None, **kwargs):
        if edge_attr is None:
            raise ValueError("BiGCN requires edge_attr with direction flag in column 1")
        assert edge_attr.shape[1] >= 2, f"BiGCN expects edge_attr with ≥2 columns, got {edge_attr.shape[1]}"
        td_mask = edge_attr[:, 1].bool()
        bu_mask = ~td_mask
        td_edge_index = edge_index[:, td_mask]
        bu_edge_index = edge_index[:, bu_mask]

        td, bu = x, x
        for td_conv, bu_conv, td_bn, bu_bn in zip(self.td_convs, self.bu_convs, self.td_bns, self.bu_bns):
            td = self.dropout(torch.relu(td_bn(td_conv(td, td_edge_index))))
            bu = self.dropout(torch.relu(bu_bn(bu_conv(bu, bu_edge_index))))

        td_g = global_mean_pool(td, batch)
        bu_g = global_mean_pool(bu, batch)
        g = torch.cat([td_g, bu_g], dim=-1)
        if self.use_text:
            t = self.text_dropout(self.text_norm(self.text_proj(root_text)))
            g = torch.cat([g, t], dim=-1)
        return self.head(g)
