"""Run all experiments: GCN, GAT, GPS on Twitter15 and Twitter16.

Usage (from project root):
    python scripts/run_all.py

Results are written to results/ as JSON files.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import CascadeDataset
from models.gcn import GCNClassifier
from models.gat import GATClassifier
from models.bigcn import BiGCNClassifier
from models.gps import GPSClassifier
from training.trainer import run_experiment

DATA_ROOT = "Twitter15_16_dataset-main"
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 200

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

os.makedirs("results", exist_ok=True)

ds15 = CascadeDataset(root=DATA_ROOT, name="twitter15")
ds16 = CascadeDataset(root=DATA_ROOT, name="twitter16")
IN_CHANNELS = ds15[0].x.shape[1]
EDGE_DIM = ds15[0].edge_attr.shape[1]
assert ds16[0].x.shape[1] == IN_CHANNELS and ds16[0].edge_attr.shape[1] == EDGE_DIM, \
    "ds15 and ds16 feature dimensions differ — check dataset processing"
print(f"Twitter15: {len(ds15)} graphs  Twitter16: {len(ds16)} graphs  in_channels={IN_CHANNELS}")

GCN_KWARGS   = dict(in_channels=IN_CHANNELS, hidden_channels=128, num_layers=3, dropout=0.1)
GAT_KWARGS   = dict(in_channels=IN_CHANNELS, hidden_channels=128, num_layers=3, heads=4, dropout=0.1)
BIGCN_KWARGS = dict(in_channels=IN_CHANNELS, hidden_channels=128, num_layers=3, dropout=0.1)
GPS_KWARGS   = dict(in_channels=IN_CHANNELS, hidden_channels=128, num_layers=4, heads=4, dropout=0.1, edge_dim=EDGE_DIM)

EXPERIMENTS = [
    ("GCN",   GCNClassifier,   GCN_KWARGS,   ds15, "results/gcn_twitter15.json",   dict(epochs=EPOCHS, warmup_ratio=0.1)),
    ("GCN",   GCNClassifier,   GCN_KWARGS,   ds16, "results/gcn_twitter16.json",   dict(epochs=EPOCHS, warmup_ratio=0.1)),
    ("GAT",   GATClassifier,   GAT_KWARGS,   ds15, "results/gat_twitter15.json",   dict(epochs=EPOCHS, warmup_ratio=0.1)),
    ("GAT",   GATClassifier,   GAT_KWARGS,   ds16, "results/gat_twitter16.json",   dict(epochs=EPOCHS, warmup_ratio=0.1)),
    ("BiGCN", BiGCNClassifier, BIGCN_KWARGS, ds15, "results/bigcn_twitter15.json", dict(epochs=EPOCHS, warmup_ratio=0.1)),
    ("BiGCN", BiGCNClassifier, BIGCN_KWARGS, ds16, "results/bigcn_twitter16.json", dict(epochs=EPOCHS, warmup_ratio=0.1)),
    ("GPS",   GPSClassifier,   GPS_KWARGS,   ds15, "results/gps_twitter15.json",   dict(epochs=EPOCHS, lr=1e-3, weight_decay=0.05, warmup_ratio=0.1, patience=40, lap_pe_sign_flip=True, max_nodes_per_batch=8192)),
    ("GPS",   GPSClassifier,   GPS_KWARGS,   ds16, "results/gps_twitter16.json",   dict(epochs=EPOCHS, lr=1e-3, weight_decay=0.05, warmup_ratio=0.1, patience=40, lap_pe_sign_flip=True, max_nodes_per_batch=8192)),
]

for model_name, model_cls, model_kwargs, dataset, out_path, extra_kwargs in EXPERIMENTS:
    ds_name = "Twitter15" if dataset is ds15 else "Twitter16"
    print(f"\n{'='*50}\n{model_name} — {ds_name}\n{'='*50}")
    res = run_experiment(
        model_cls, model_kwargs, dataset, SEEDS,
        device=DEVICE,
        **extra_kwargs,
    )
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  → saved to {out_path}")

# Print final summary table
print("\n" + "=" * 65)
print(f"{'Model':<10} {'Dataset':<14} {'Acc mean±std':>16}  {'Macro-F1 mean±std':>18}")
print("-" * 65)
for model_name, _, _, dataset, out_path, _ in EXPERIMENTS:
    ds_name = "Twitter15" if dataset is ds15 else "Twitter16"
    with open(out_path) as f:
        res = json.load(f)
    print(f"{model_name:<10} {ds_name:<14} {res['test_acc_mean']:.3f} ± {res['test_acc_std']:.3f}        {res['test_f1_mean']:.3f} ± {res['test_f1_std']:.3f}")
print("=" * 65)
print("(5 seeds, 60/20/20 stratified split, best val-F1 checkpoint)")
