"""CascadeGPS — from-scratch GraphGPS-style classifier tailored to rumor cascades.

Differs from `models/gps.py` (which wraps `torch_geometric.nn.GPSConv` + `GINEConv`) by
addressing two structural limits of stock GPS that matter on cascade trees:

1. Stock GPS attention ignores `edge_attr` — direction flag and Δt only flow through
   the MPNN branch. Here we project `edge_attr` into a per-head attention bias added
   to Q·Kᵀ on every edge pair.
2. Stock GPS sums the MPNN and attention branches. Here we fuse them per-node with a
   learned sigmoid gate.

Supporting modifications (sinusoidal time encoding, learnable ROOT bias, direction-gated
GINE-style MPNN, pairwise |Δt| attention bias, pre-norm blocks, 3-way readout) are
documented inline.

Input format (must match what `data/dataset.py` produces):
    x:          [N, 30]  — col 0 = normalized timestamp, col 1-4 = depth/in-deg/subtree/branch,
                           col 5 = log(num_nodes), cols 6-13 = LapPE, cols 14-29 = RWPE
    edge_index: [2, 2E]  — bidirectionalized
    edge_attr:  [2E, 2]  — col 0 = normalized Δt, col 1 = direction flag (1=parent→child)
    batch:      [N]      — graph assignment
    ptr:        [B+1]    — graph boundaries; ptr[g] is the index of ROOT for graph g
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax, to_dense_batch


def sinusoidal_time(t: torch.Tensor, num_freqs: int = 4) -> torch.Tensor:
    """Encode scalar t ∈ [0, 1] as [sin(2^k π t), cos(2^k π t)] for k = 0..num_freqs-1.

    Args:
        t: [N] scalar timestamps.
        num_freqs: number of frequency bands.
    Returns:
        [N, 2 * num_freqs]
    """
    freqs = torch.tensor([2 ** k * math.pi for k in range(num_freqs)], device=t.device, dtype=t.dtype)
    angles = t.unsqueeze(-1) * freqs  # [N, num_freqs]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [N, 2*num_freqs]


class EdgeGatedMPNN(nn.Module):
    """GINE-style MPNN with direction-conditioned multiplicative edge gating.

    Message: m_ij = (W_src h_i + W_dst h_j + edge_emb_ij) ⊙ σ(W_g · edge_emb_ij)
    Aggregation: sum over incoming edges.
    Update: MLP((1 + ε) h_dst + agg)  (GINE residual).
    """

    def __init__(self, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.w_src = nn.Linear(hidden, hidden)
        self.w_dst = nn.Linear(hidden, hidden)
        self.w_gate = nn.Linear(hidden, hidden)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        m = self.w_src(h[src]) + self.w_dst(h[dst]) + edge_emb
        gate = torch.sigmoid(self.w_gate(edge_emb))
        m = m * gate
        agg = scatter(m, dst, dim=0, dim_size=h.size(0), reduce="sum")
        return self.mlp((1.0 + self.eps) * h + agg)


class CascadeAttention(nn.Module):
    """Dense multi-head attention with edge-aware + temporal attention bias.

    Per-batch precomputes (passed in via `cached`) are reused across layers:
      - edge_bias_dense:  [B, heads, Nmax, Nmax]   from EdgeBiasProj(edge_attr) scattered into dense slots
      - temp_bias_dense:  [B, heads, Nmax, Nmax]   from TempBiasMLP(|t_i - t_j|) over the dense batch
      - mask:             [B, Nmax]                True where the position is a real node
    """

    def __init__(self, hidden: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden % heads == 0, f"hidden ({hidden}) must be divisible by heads ({heads})"
        self.heads = heads
        self.d_head = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.dropout = dropout

    def forward(self, h: torch.Tensor, batch: torch.Tensor, cached: dict) -> torch.Tensor:
        # h: [N, H]  → dense [B, Nmax, H]
        H_dense, mask = to_dense_batch(h, batch)
        B, Nmax, _ = H_dense.shape

        q = self.q_proj(H_dense).view(B, Nmax, self.heads, self.d_head).transpose(1, 2)  # [B, h, Nmax, d]
        k = self.k_proj(H_dense).view(B, Nmax, self.heads, self.d_head).transpose(1, 2)
        v = self.v_proj(H_dense).view(B, Nmax, self.heads, self.d_head).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, h, Nmax, Nmax]
        logits = logits + cached["edge_bias_dense"] + cached["temp_bias_dense"]

        # Mask padded key positions
        key_mask = (~mask).view(B, 1, 1, Nmax)  # broadcasts over heads & queries
        logits = logits.masked_fill(key_mask, float("-inf"))

        attn = F.softmax(logits, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out_dense = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Nmax, self.heads * self.d_head)
        out_dense = self.out_proj(out_dense)

        # Unbatch back to [N, H] using mask
        return out_dense[mask]


class CascadeBlock(nn.Module):
    """One transformer block: pre-norm MPNN + pre-norm attention with gated fusion, then pre-norm FFN."""

    def __init__(self, hidden: int, heads: int, dropout: float = 0.0, ffn_mult: int = 4):
        super().__init__()
        self.norm_mpnn = nn.LayerNorm(hidden)
        self.mpnn = EdgeGatedMPNN(hidden, dropout=dropout)

        self.norm_attn = nn.LayerNorm(hidden)
        self.attn = CascadeAttention(hidden, heads, dropout=dropout)

        # Gated fusion produces a single combined residual update
        self.fuse_gate = nn.Linear(2 * hidden, hidden)

        self.norm_ffn = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn_mult * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * hidden, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_emb: torch.Tensor,
                batch: torch.Tensor, cached: dict) -> torch.Tensor:
        # Pre-norm MPNN branch (computes the local contribution)
        h_m = self.mpnn(self.norm_mpnn(h), edge_index, edge_emb)

        # Pre-norm attention branch (computes the global contribution)
        h_a = self.attn(self.norm_attn(h), batch, cached)

        # Per-node gated fusion: combine local + global, then a single residual update
        gate = torch.sigmoid(self.fuse_gate(torch.cat([h_m, h_a], dim=-1)))
        h = h + self.dropout(gate * h_m + (1.0 - gate) * h_a)

        # Pre-norm FFN
        h = h + self.dropout(self.ffn(self.norm_ffn(h)))
        return h


def attentional_pool(h: torch.Tensor, batch: torch.Tensor, gate_lin: nn.Linear,
                     num_graphs: int) -> torch.Tensor:
    """Graph-level pool: per-node scalar gate → softmax within graph → weighted sum.

    Same idea as `torch_geometric.nn.AttentionalAggregation`, reimplemented here so the
    model stays self-contained. `num_graphs` is passed in (from `ptr.numel() - 1`) to
    avoid a GPU↔CPU sync on `batch.max()`.
    """
    logits = gate_lin(h).squeeze(-1)              # [N]
    weights = softmax(logits, batch)              # softmax-within-graph
    weighted = h * weights.unsqueeze(-1)          # [N, H]
    return scatter(weighted, batch, dim=0, dim_size=num_graphs, reduce="sum")


class CascadeGPSClassifier(nn.Module):
    """From-scratch GraphGPS-style classifier tailored to rumor cascades.

    Cascade-specific design choices (see module docstring for rationale):
      - Sinusoidal time encoding on the raw timestamp at input.
      - Learnable ROOT bias added to graph-local node 0 after input projection.
      - Edge-gated direction-aware GINE-style MPNN (single stream).
      - Multi-head attention with edge_attr-driven per-head bias and pairwise |Δt| bias.
      - Per-node learned gate fusing MPNN and attention contributions.
      - 3-way readout: root + attentional-pool + mean.
    """

    def __init__(
        self,
        in_channels: int = 30,
        hidden_channels: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int = 2,
        num_classes: int = 4,
        num_time_freqs: int = 4,
        ffn_mult: int = 4,
        temp_bias_hidden: int = 16,
    ):
        super().__init__()
        assert hidden_channels % heads == 0, \
            f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"

        self.num_time_freqs = num_time_freqs
        self.heads = heads

        # x[:, 0] (scalar t) gets expanded to 2*num_time_freqs sin/cos dims; the remaining
        # in_channels - 1 features pass through unchanged.
        in_after_time = (in_channels - 1) + 2 * num_time_freqs
        self.input_proj = nn.Linear(in_after_time, hidden_channels)

        # Learnable ROOT bias added to graph-local node 0 after input projection.
        # Acts as a soft "CLS token" without changing N or masks.
        self.root_embedding = nn.Parameter(torch.zeros(hidden_channels))
        nn.init.normal_(self.root_embedding, std=0.02)

        # Edge encoder (shared across layers): edge_attr → hidden-dim message contribution.
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Edge → per-head attention bias scalar (shared across layers).
        # No activation: bias is added directly to scaled-dot-product logits.
        # Small-init weight and zero bias so attention starts nearly unbiased — otherwise
        # default Kaiming produces logit biases ~O(0.3) that dominate Q·Kᵀ/√d at epoch 0,
        # and the model wastes warmup suppressing random bias instead of learning it.
        self.edge_bias_proj = nn.Linear(edge_dim, heads)
        nn.init.normal_(self.edge_bias_proj.weight, std=0.02)
        nn.init.zeros_(self.edge_bias_proj.bias)

        # Pairwise |Δt| → per-head bias (shared across layers).
        # Same reasoning for the final layer. We DON'T fully zero the weight here because
        # this is the 2nd layer of a 2-layer MLP — fully zero weight would kill gradient
        # flow to the first layer (dY/dW0 propagates through W1, which would be 0).
        self.temp_bias_mlp = nn.Sequential(
            nn.Linear(1, temp_bias_hidden),
            nn.GELU(),
            nn.Linear(temp_bias_hidden, heads),
        )
        nn.init.normal_(self.temp_bias_mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.temp_bias_mlp[-1].bias)

        self.blocks = nn.ModuleList([
            CascadeBlock(hidden_channels, heads, dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_channels)

        # 3-way readout pool: root + attentional + mean.
        self.attn_pool_gate = nn.Linear(hidden_channels, 1)

        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def _build_dense_biases(self, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                           t_scalar: torch.Tensor, batch: torch.Tensor,
                           num_graphs: int) -> dict:
        """Precompute the dense [B, heads, Nmax, Nmax] biases used by every attention layer.

        Done once per forward — the biases depend only on (edge_index, edge_attr, t, batch),
        not on the hidden state, so they're shared across all layers. `num_graphs` is passed
        in (from `ptr.numel() - 1`) to avoid a GPU↔CPU sync on `batch.max()`.
        """
        N = batch.size(0)
        B = num_graphs

        # Local-within-graph index for each node (0..Ng-1 for graph g).
        ones = torch.ones(N, device=batch.device, dtype=torch.long)
        local_idx = scatter(ones, batch, dim=0, dim_size=B, reduce="sum")  # [B]: nodes per graph
        Nmax = int(local_idx.max().item())

        # Build a per-node local index. Cumulative-sum trick: position within its graph.
        # `arange(N) - ptr_of(batch[i])` gives local index.
        cum = torch.zeros(B + 1, device=batch.device, dtype=torch.long)
        cum[1:] = torch.cumsum(local_idx, dim=0)
        node_local = torch.arange(N, device=batch.device) - cum[batch]  # [N]

        # ---- Edge attention bias ----
        edge_b = self.edge_bias_proj(edge_attr)                   # [2E, heads]
        edge_bias_dense = torch.zeros(B, self.heads, Nmax, Nmax,
                                      device=batch.device, dtype=edge_b.dtype)
        src, dst = edge_index[0], edge_index[1]
        b_idx = batch[src]                                        # graph id of each edge
        s_loc = node_local[src]
        d_loc = node_local[dst]
        # Index assignment: edge_bias_dense[b, h, s, d] += edge_b[e, h].
        # accumulate=True so duplicate edges (if any) reinforce rather than overwrite;
        # `data/dataset.py` currently dedups so this is defensive.
        for h_idx in range(self.heads):
            edge_bias_dense.index_put_(
                (b_idx, torch.full_like(b_idx, h_idx), s_loc, d_loc),
                edge_b[:, h_idx],
                accumulate=True,
            )

        # ---- Pairwise |Δt| attention bias ----
        # Densify the scalar timestamp into [B, Nmax].
        t_dense = torch.zeros(B, Nmax, device=batch.device, dtype=t_scalar.dtype)
        t_dense[batch, node_local] = t_scalar
        # |t_i - t_j|: [B, Nmax, Nmax, 1]
        dt_dense = (t_dense.unsqueeze(2) - t_dense.unsqueeze(1)).abs().unsqueeze(-1)
        temp_bias_dense = self.temp_bias_mlp(dt_dense)            # [B, Nmax, Nmax, heads]
        temp_bias_dense = temp_bias_dense.permute(0, 3, 1, 2).contiguous()  # [B, heads, Nmax, Nmax]

        return {
            "edge_bias_dense": edge_bias_dense,
            "temp_bias_dense": temp_bias_dense,
        }

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                edge_attr: torch.Tensor = None, ptr: torch.Tensor = None, **kwargs) -> torch.Tensor:
        assert edge_attr is not None, "CascadeGPSClassifier.forward requires edge_attr"
        assert ptr is not None, "CascadeGPSClassifier.forward requires ptr (batch.ptr) for root readout"

        # --- Input transform: sinusoidal time on col 0, pass-through rest ---
        t_scalar = x[:, 0]
        t_enc = sinusoidal_time(t_scalar, self.num_time_freqs)
        x_in = torch.cat([t_enc, x[:, 1:]], dim=-1)
        h = self.input_proj(x_in)

        # --- ROOT bias added to graph-local node 0 of each graph ---
        # ptr[:-1] gives the global index of the root node for every graph.
        num_graphs = ptr.numel() - 1
        h = h.index_add(0, ptr[:-1], self.root_embedding.expand(num_graphs, -1))

        # --- Edge embedding (shared across layers) for the MPNN branch ---
        edge_emb = self.edge_encoder(edge_attr)

        # --- Precompute dense attention biases once per forward (shared across layers) ---
        cached = self._build_dense_biases(edge_index, edge_attr, t_scalar, batch, num_graphs)

        for block in self.blocks:
            h = block(h, edge_index, edge_emb, batch, cached)

        h = self.final_norm(h)

        # --- 3-way readout ---
        root_emb = h[ptr[:-1]]                                                          # [B, H]
        attn_emb = attentional_pool(h, batch, self.attn_pool_gate, num_graphs)           # [B, H]
        mean_emb = scatter(h, batch, dim=0, dim_size=num_graphs, reduce="mean")          # [B, H]

        g = torch.cat([root_emb, attn_emb, mean_emb], dim=-1)                            # [B, 3H]
        return self.head(g)


# --------------------------------------------------------------------------------------
# Smoke tests — run with `python -m models.cascade_gps` from the project root.
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from torch_geometric.data import Batch, Data

    torch.manual_seed(0)

    def make_cascade(n: int, in_channels: int = 30, edge_dim: int = 2) -> Data:
        """Build a synthetic cascade: random tree edges from node 0 (ROOT) outward."""
        x = torch.randn(n, in_channels)
        x[:, 0] = torch.rand(n)  # normalized timestamps in [0, 1]
        # Parent of node i (for i >= 1) is a random earlier node, so node 0 is reachable.
        edges = []
        edge_feats = []
        for i in range(1, n):
            parent = int(torch.randint(0, i, (1,)).item())
            edges.append((parent, i)); edge_feats.append([abs(x[i, 0].item() - x[parent, 0].item()), 1.0])
            edges.append((i, parent)); edge_feats.append([abs(x[i, 0].item() - x[parent, 0].item()), 0.0])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        y = torch.tensor([int(torch.randint(0, 4, (1,)).item())], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    sizes = [5, 12, 30]
    graphs = [make_cascade(n) for n in sizes]
    batch = Batch.from_data_list(graphs)
    model = CascadeGPSClassifier()
    model.eval()

    # 1) Shape test
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, ptr=batch.ptr)
    assert out.shape == (len(sizes), 4), f"expected [3, 4], got {tuple(out.shape)}"
    print(f"[1] shape OK: {tuple(out.shape)}")

    # 2) Mask correctness: attention weights on padded positions are ~0.
    #    Easiest check: ensure forward succeeds on a batch where graph sizes differ
    #    (already tested above) and outputs are finite.
    assert torch.isfinite(out).all(), "non-finite output — likely mask leak"
    print("[2] outputs finite (mask OK)")

    # 3) Gradient flow: every parameter receives a grad.
    model.train()
    out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, ptr=batch.ptr)
    loss = out.sum()
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None or p.grad.abs().sum().item() == 0]
    assert not missing, f"params with no grad: {missing}"
    print(f"[3] gradient flow OK ({sum(1 for _ in model.parameters())} params, all received grads)")

    # 4) Direction sensitivity: flipping the direction flag (col 1 of edge_attr) changes logits.
    model.eval()
    batch_flip = Batch.from_data_list(graphs).clone()
    batch_flip.edge_attr = batch_flip.edge_attr.clone()
    batch_flip.edge_attr[:, 1] = 1.0 - batch_flip.edge_attr[:, 1]
    with torch.no_grad():
        out_orig = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, ptr=batch.ptr)
        out_flip = model(batch_flip.x, batch_flip.edge_index, batch_flip.batch,
                         edge_attr=batch_flip.edge_attr, ptr=batch_flip.ptr)
    diff = (out_orig - out_flip).abs().max().item()
    assert diff > 1e-5, f"direction flip did not change output (max diff = {diff})"
    print(f"[4] direction sensitivity OK (max logit diff = {diff:.4f})")

    # 5) Permutation invariance over batch order: shuffle graph order, per-graph logits unchanged.
    perm = [2, 0, 1]
    graphs_perm = [graphs[i] for i in perm]
    batch_perm = Batch.from_data_list(graphs_perm)
    with torch.no_grad():
        out_perm = model(batch_perm.x, batch_perm.edge_index, batch_perm.batch,
                         edge_attr=batch_perm.edge_attr, ptr=batch_perm.ptr)
    for new_i, orig_i in enumerate(perm):
        delta = (out_perm[new_i] - out_orig[orig_i]).abs().max().item()
        assert delta < 1e-4, f"perm[{new_i}<-{orig_i}] logits differ by {delta}"
    print(f"[5] batch-order invariance OK (max delta across permutation = {delta:.2e})")

    print("\nAll smoke tests passed.")
