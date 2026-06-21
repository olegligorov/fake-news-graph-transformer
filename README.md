# Fake News Detection via Graph Transformers

Classifies Twitter rumour cascades (true / false / unverified / non-rumor) using GraphGPS — a hybrid local MPNN + global Transformer. Built to demonstrate that global attention outperforms local GNNs (GCN, GAT) on long-range cascade patterns.

Dataset: Twitter15 and Twitter16. No tweet text — structure and timing only.

---

## Setup

**Requirements:** Python 3.11, CUDA-capable GPU recommended (tested on RTX 5070 8 GB).

```bash
# Clone and enter
git clone <repo-url>
cd fake-news-graph-transformer

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Data

Download the [Twitter15/16 dataset](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip) and extract it so the layout is:

```
Twitter15_16_dataset-main/
  twitter15/
    label.txt
    tree/
      <cascade_id>.txt
      ...
  twitter16/
    label.txt
    tree/
      ...
```

Place this folder at the project root (sibling to `data/`, `models/`, `notebooks/`).

---

## Project structure

```
data/
  dataset.py      # CascadeDataset — parses cascade files into PyG Data objects
  features.py     # Per-node structural features (timestamp, depth, degree, subtree size)
  transforms.py   # Laplacian PE + Random Walk PE appended to node features
models/
  gcn.py          # GCN baseline
  gat.py          # GAT baseline
training/
  trainer.py      # Training loop, stratified split, run_experiment helper
notebooks/
  01_data_exploration.ipynb   # Dataset stats and cascade visualisation
  02_baselines.ipynb          # GCN and GAT results over 5 seeds
```

---

## Running

### Data exploration

```bash
cd notebooks
jupyter lab
# open 01_data_exploration.ipynb
```

The first run processes raw cascade files and caches them to `Twitter15_16_dataset-main/twitter15/processed/data.pt` (and twitter16). Subsequent runs load from cache.

### Baselines (GCN + GAT)

```bash
# open 02_baselines.ipynb in JupyterLab
```

Trains each model for 200 epochs across 5 seeds on a fixed 60/20/20 stratified split. Reports mean ± std macro-F1 and accuracy on both datasets.

### Using the dataset in a script

```python
from data.dataset import CascadeDataset

ds = CascadeDataset(root="Twitter15_16_dataset-main", name="twitter15")
data = ds[0]
# data.x          [N, 29]  — 5 structural + 8 Laplacian PE + 16 RWPE
# data.edge_index [2, E]
# data.edge_attr  [E, 1]   — normalised timestamp delta
# data.y          [1]      — 0=false, 1=true, 2=unverified, 3=non-rumor
```

---

## Node features

Each node receives 29 features: 5 structural (normalised per cascade) + 8 Laplacian eigenvector PE + 16 random-walk PE.

| # | Feature | Description |
|---|---------|-------------|
| 0 | timestamp | Hours since root tweet, normalised |
| 1 | depth | Depth in cascade tree (ROOT = 0) |
| 2 | in-degree | Number of parents |
| 3 | subtree size | Number of descendants |
| 4 | branching factor | Out-degree |
| 5–12 | Laplacian PE | k=8 eigenvectors |
| 13–28 | Random Walk PE | Walk length 16 |
