"""Per-cascade node feature computation.

All five features are normalized to [0, 1] within the cascade to avoid scale issues.
"""

from collections import defaultdict, deque

import torch


def compute_features(edge_index: torch.Tensor, timestamps: list[float], num_nodes: int) -> torch.Tensor:
    """Return float32 tensor of shape [num_nodes, 5].

    Features per node (in order):
        0: timestamp (hours since root, normalized)
        1: depth in cascade tree (ROOT = 0, normalized)
        2: in-degree (normalized)
        3: subtree size (normalized)
        4: branching factor / out-degree (normalized)

    Args:
        edge_index: [2, E] directed edges (parent → child), int64
        timestamps: list of float timestamps per node, length num_nodes
        num_nodes: total node count including ROOT (node 0)
    """
    # --- adjacency lists (deduplicated — dataset.py already drops dupes, but be safe) ---
    children: list[set[int]] = [set() for _ in range(num_nodes)]
    in_deg = [0] * num_nodes

    if edge_index.numel() > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst):
            if v not in children[u]:
                children[u].add(v)
                in_deg[v] += 1

    # --- depth via BFS from ROOT (node 0) ---
    depth = [-1] * num_nodes
    depth[0] = 0
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for child in children[node]:
            if depth[child] == -1:
                depth[child] = depth[node] + 1
                queue.append(child)
    # nodes unreachable from ROOT (disconnected due to parse issues) get depth 0
    depth = [d if d >= 0 else 0 for d in depth]

    # --- subtree sizes via post-order DFS ---
    subtree_size = [1] * num_nodes
    visited = [False] * num_nodes
    stack = [(0, False)]
    while stack:
        node, returning = stack.pop()
        if returning:
            for child in children[node]:  # children is a set — no duplicates
                subtree_size[node] += subtree_size[child]
        else:
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, True))
            for child in children[node]:
                if not visited[child]:
                    stack.append((child, False))

    out_deg = [len(c) for c in children]  # children is a set so no duplicates

    # --- assemble raw features ---
    ts = torch.tensor(timestamps, dtype=torch.float32)
    dep = torch.tensor(depth, dtype=torch.float32)
    ind = torch.tensor(in_deg, dtype=torch.float32)
    sub = torch.tensor(subtree_size, dtype=torch.float32)
    bra = torch.tensor(out_deg, dtype=torch.float32)

    # --- per-cascade min-max normalization ---
    def _norm(t: torch.Tensor) -> torch.Tensor:
        lo, hi = t.min(), t.max()
        if hi - lo < 1e-8:
            return torch.zeros_like(t)
        return (t - lo) / (hi - lo)

    feats = torch.stack([_norm(ts), _norm(dep), _norm(ind), _norm(sub), _norm(bra)], dim=1)
    return feats
