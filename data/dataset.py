"""Twitter15/16 cascade graph dataset.

Usage:
    from data.dataset import CascadeDataset
    ds = CascadeDataset(root="/path/to/Twitter15_16_dataset-main", name="twitter15")
    data = ds[0]  # PyG Data object
    # data.x          shape [N, 29]  (5 structural + 8 lap_pe + 16 rw_pe)
    # data.edge_index shape [2, 2E]  (directed + reverse edges for bidirectional message passing)
    # data.edge_attr  shape [2E, 1]  (normalized timestamp delta, same value for both directions)
    # data.y          shape [1]      (0=false, 1=true, 2=unverified, 3=non-rumor)
"""

import re
import warnings
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset

from data.features import compute_features
from data.transforms import add_positional_encodings

LABEL_MAP = {"false": 0, "true": 1, "unverified": 2, "non-rumor": 3}

# Matches lines like:  ['user_id', 'tweet_id', 'timestamp']->['user_id', 'tweet_id', 'timestamp']
_LINE_RE = re.compile(
    r"\['(.+?)',\s*'(.+?)',\s*'(.+?)'\]->\['(.+?)',\s*'(.+?)',\s*'(.+?)'\]"
)


def _parse_cascade(path: Path) -> tuple[list[tuple], list[tuple]]:
    """Return (edges, nodes) where nodes are (user_id, timestamp).

    edges is a list of (src_uid, dst_uid) strings.
    """
    edges: list[tuple[str, str]] = []
    timestamps: dict[str, float] = {}

    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = _LINE_RE.match(raw)
        if not m:
            continue
        src_uid, _, src_ts, dst_uid, _, dst_ts = m.groups()
        timestamps[src_uid] = float(src_ts)
        timestamps[dst_uid] = float(dst_ts)
        edges.append((src_uid, dst_uid))

    return edges, timestamps


def _build_graph(cascade_id: str, label: int, tree_dir: Path) -> Data | None:
    path = tree_dir / f"{cascade_id}.txt"
    if not path.exists():
        warnings.warn(f"Missing cascade file: {path}", stacklevel=2)
        return None

    try:
        raw_edges, timestamps = _parse_cascade(path)
    except Exception as exc:
        warnings.warn(f"Failed to parse {path}: {exc}", stacklevel=2)
        return None

    if not timestamps:
        warnings.warn(f"Empty cascade: {path}", stacklevel=2)
        return None

    # Node 0 is always ROOT; remaining nodes in insertion order
    uid_to_idx: dict[str, int] = {"ROOT": 0}
    for uid in timestamps:
        if uid != "ROOT" and uid not in uid_to_idx:
            uid_to_idx[uid] = len(uid_to_idx)

    num_nodes = len(uid_to_idx)
    ts_raw = [0.0] * num_nodes
    for uid, idx in uid_to_idx.items():
        ts_raw[idx] = timestamps.get(uid, 0.0)

    if not raw_edges:
        warnings.warn(f"No edges in cascade: {path}", stacklevel=2)
        return None

    seen_edges: set[tuple[int, int]] = set()
    src_list, dst_list, delta_list = [], [], []
    for src_uid, dst_uid in raw_edges:
        if src_uid not in uid_to_idx or dst_uid not in uid_to_idx:
            continue
        si, di = uid_to_idx[src_uid], uid_to_idx[dst_uid]
        if si == di or (si, di) in seen_edges:  # drop self-loops and duplicates
            continue
        seen_edges.add((si, di))
        src_list.append(si)
        dst_list.append(di)
        delta_list.append(abs(ts_raw[di] - ts_raw[si]))

    if not src_list:
        warnings.warn(f"All edges filtered in cascade: {path}", stacklevel=2)
        return None

    # Directed edges (parent → child) — used for structural feature computation
    edge_index_dir = torch.tensor([src_list, dst_list], dtype=torch.long)
    raw_delta = torch.tensor(delta_list, dtype=torch.float32).unsqueeze(1)
    max_delta = raw_delta.max()
    edge_attr_dir = raw_delta / (max_delta + 1e-8)

    x = compute_features(edge_index_dir, ts_raw, num_nodes)

    # Add reverse edges so leaves can propagate messages upward during GNN message passing.
    # 89% of nodes in these cascades are leaves; without reverse edges they are silent.
    # Direction is already encoded in structural node features (depth, timestamp, subtree_size).
    ei_full = torch.cat([edge_index_dir, edge_index_dir.flip(0)], dim=1)
    ea_full = torch.cat([edge_attr_dir, edge_attr_dir], dim=0)
    # Deduplicate (handles raw files that already contain both A→B and B→A)
    seen: set[tuple[int, int]] = set()
    keep_mask = []
    for i in range(ei_full.shape[1]):
        k = (ei_full[0, i].item(), ei_full[1, i].item())
        keep_mask.append(k not in seen)
        seen.add(k)
    idx = torch.tensor(keep_mask)
    edge_index = ei_full[:, idx]
    edge_attr = ea_full[idx]
    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes)
    data = add_positional_encodings(data)
    return data


class CascadeDataset(InMemoryDataset):
    """PyG InMemoryDataset for Twitter15 or Twitter16 cascade graphs.

    Args:
        root: path to the folder containing twitter15/ and twitter16/ subdirectories
        name: "twitter15" or "twitter16"
    """

    def __init__(self, root: str, name: str):
        name = name.lower()
        if name not in ("twitter15", "twitter16"):
            raise ValueError(f"name must be 'twitter15' or 'twitter16', got '{name}'")
        self.name = name
        super().__init__(root=root)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return str(Path(self.root) / self.name)

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / self.name / "processed")

    @property
    def raw_file_names(self) -> list[str]:
        return ["label.txt"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    def download(self):
        pass  # data ships with the repo

    def process(self):
        label_path = Path(self.raw_dir) / "label.txt"
        tree_dir = Path(self.raw_dir) / "tree"

        labels: dict[str, int] = {}
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            label_str, cascade_id = line.split(":", 1)
            labels[cascade_id] = LABEL_MAP[label_str]

        data_list: list[Data] = []
        for cascade_id, label in labels.items():
            graph = _build_graph(cascade_id, label, tree_dir)
            if graph is not None:
                data_list.append(graph)

        self.save(data_list, self.processed_paths[0])
