import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from data.transforms import NODE_FEATURES_BASE, LAP_PE_DIM, NODE_FEATURES_TOTAL


def stratified_split(dataset: Dataset, train_ratio: float = 0.6, val_ratio: float = 0.2, split_seed: int = 0):
    """Return (train_idx, val_idx, test_idx) lists with stratification on y.

    split_seed is fixed across model seeds so all seeds see the same train/val/test split.
    Only model initialization varies across seeds, enabling paired comparisons.
    """
    labels = [dataset[i].y.item() for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    test_ratio = 1.0 - train_ratio - val_ratio

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=(1.0 - train_ratio), stratify=labels, random_state=split_seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_ratio / (test_ratio + val_ratio), stratify=temp_labels, random_state=split_seed
    )
    return train_idx, val_idx, test_idx


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _flip_lap_pe(batch, lap_pe_start: int = NODE_FEATURES_BASE, lap_pe_dim: int = LAP_PE_DIM):
    """Randomly flip sign of each LapPE eigenvector for all nodes in the same graph.

    LapPE eigenvectors are sign-arbitrary (v and -v are both valid). Without this
    augmentation the model overfits to the arbitrary sign choices in the dataset.
    Operates in-place on batch.x columns [lap_pe_start : lap_pe_start + lap_pe_dim].
    """
    assert batch.x.shape[1] == NODE_FEATURES_TOTAL, \
        f"Expected {NODE_FEATURES_TOTAL} node features, got {batch.x.shape[1]} — check data/transforms.py constants"
    signs = (torch.randint(0, 2, (batch.num_graphs, lap_pe_dim), device=batch.x.device) * 2 - 1).float()
    node_signs = signs[batch.batch]  # [num_nodes, lap_pe_dim]
    batch.x[:, lap_pe_start: lap_pe_start + lap_pe_dim] *= node_signs


def _make_lr_lambda(warmup_steps: int, total_steps: int):
    """Return a per-epoch LR multiplier: linear warmup then cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def _token_batches(indices: list[int], dataset: Dataset, max_nodes: int, shuffle: bool, seed: int) -> list[list[int]]:
    """Group indices into batches where total nodes per batch <= max_nodes.

    Used for GPSConv multihead attention which runs to_dense_batch(N_max^2) per batch.
    Prevents OOM when a few large cascades land in the same batch.
    """
    rng = random.Random(seed)
    if shuffle:
        indices = list(indices)
        rng.shuffle(indices)
    batches, current, current_nodes = [], [], 0
    for idx in indices:
        n = dataset[idx].num_nodes
        if current and current_nodes + n > max_nodes:
            batches.append(current)
            current, current_nodes = [], 0
        current.append(idx)
        current_nodes += n
    if current:
        batches.append(current)
    return batches


def _make_token_loader(indices: list[int], dataset: Dataset, max_nodes: int, shuffle: bool, seed: int) -> list:
    """Group indices into node-budget batches and pre-collate each group into a Batch.

    Returns a plain list of Batch objects. train_epoch/eval_epoch iterate with
    `for batch in loader` which works on any iterable — no DataLoader wrapper needed
    since the batching is already done here.
    """
    from torch_geometric.data import Batch
    batches = _token_batches(indices, dataset, max_nodes, shuffle=shuffle, seed=seed)
    return [Batch.from_data_list([dataset[i] for i in b]) for b in batches]


def train_epoch(model: nn.Module, loader, optimizer, criterion, device, lap_pe_sign_flip: bool = False) -> float:
    model.train()
    total_loss = 0.0
    n_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        if lap_pe_sign_flip:
            _flip_lap_pe(batch)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, ptr=batch.ptr, root_text=getattr(batch, "root_text", None))
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs
    return total_loss / n_graphs


@torch.no_grad()
def eval_epoch(model: nn.Module, loader, criterion, device, num_classes: int | None = None) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    n_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, ptr=batch.ptr, root_text=getattr(batch, "root_text", None))
        loss = criterion(out, batch.y.view(-1))
        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y.view(-1).cpu().tolist())
        n_graphs += batch.num_graphs

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    labels_axis = list(range(num_classes)) if num_classes is not None else None
    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=labels_axis, zero_division=0).tolist()
    cm = confusion_matrix(all_labels, all_preds, labels=labels_axis).tolist()
    return {
        "loss": total_loss / n_graphs,
        "acc": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


def run_experiment(
    model_cls,
    model_kwargs: dict,
    dataset: Dataset,
    seeds: list[int],
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    patience: int | None = None,
    warmup_ratio: float = 0.1,
    lap_pe_sign_flip: bool = False,
    max_nodes_per_batch: int | None = None,
    device: str = "cpu",
    verbose: bool = True,
    results_dir: str | None = None,
    model_name: str | None = None,
    dataset_name: str | None = None,
) -> dict:
    """Train model_cls for each seed; return per-seed and aggregate results.

    Args:
        patience: epochs without val_f1 improvement before early stopping. None = disabled.
        warmup_ratio: fraction of total epochs used for linear LR warmup (cosine decay after).
        lap_pe_sign_flip: randomly flip LapPE eigenvector signs each training batch.
                          Required for GPS; not needed for GCN/GAT.
        max_nodes_per_batch: if set, use token-budget batching (cap total nodes per batch).
                             Required for GPS multihead attention to avoid OOM on large cascades.
                             Ignored when None (uses fixed batch_size instead).
        results_dir: if set, write per-seed JSONs at
                     {results_dir}/{model_name}_{dataset_name}_text-{on|off}_readout-{mean|4way}_seed{N}.json.
                     Requires model_name and dataset_name.
    """
    # Fixed split across all seeds — only model init varies
    train_idx, val_idx, test_idx = stratified_split(dataset, split_seed=0)

    # Class-weighted CE to handle label imbalance
    y_train = [dataset[i].y.item() for i in train_idx]
    num_classes = int(max(dataset[i].y.item() for i in range(len(dataset)))) + 1
    classes_present = np.unique(y_train)
    w = compute_class_weight("balanced", classes=classes_present, y=y_train)
    weights = np.ones(num_classes, dtype=np.float32)
    for c, wc in zip(classes_present, w):
        weights[int(c)] = wc
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    if results_dir is not None:
        assert model_name is not None and dataset_name is not None, \
            "results_dir requires model_name and dataset_name"
        os.makedirs(results_dir, exist_ok=True)
    use_text = bool(model_kwargs.get("use_text", False))
    readout = str(model_kwargs.get("readout", "mean"))

    seed_results = []

    for seed in seeds:
        _set_seed(seed)

        if max_nodes_per_batch is not None:
            # Token-budget batching for GPS multihead attention (avoids N_max² OOM).
            # Val/test loaders are fixed (no shuffle). Train loader is rebuilt each epoch
            # with a new seed so graph groupings reshuffle across epochs.
            val_loader  = _make_token_loader(val_idx,  dataset, max_nodes_per_batch, shuffle=False, seed=seed)
            test_loader = _make_token_loader(test_idx, dataset, max_nodes_per_batch, shuffle=False, seed=seed)
            use_token_batching = True
        else:
            train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader([dataset[i] for i in val_idx],   batch_size=batch_size)
            test_loader  = DataLoader([dataset[i] for i in test_idx],  batch_size=batch_size)
            use_token_batching = False

        model = model_cls(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        total_steps = epochs
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _make_lr_lambda(warmup_steps, total_steps))

        best_val_f1 = -float("inf")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}  # snapshot before training
        epochs_no_improve = 0

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            if use_token_batching:
                # Rebuild each epoch so graph groupings reshuffle (different seed per epoch)
                train_loader = _make_token_loader(train_idx, dataset, max_nodes_per_batch, shuffle=True, seed=seed * 10000 + epoch)
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, lap_pe_sign_flip=lap_pe_sign_flip)
            scheduler.step()
            val_metrics = eval_epoch(model, val_loader, criterion, device, num_classes=num_classes)

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and epoch % 10 == 0:
                print(f"  seed={seed} epoch={epoch:3d}  train_loss={train_loss:.4f}  val_f1={val_metrics['macro_f1']:.4f}  best={best_val_f1:.4f}")

            if patience is not None and epochs_no_improve >= patience:
                if verbose:
                    print(f"  seed={seed} early stop at epoch {epoch}")
                break

        train_time_sec = time.time() - t0
        model.load_state_dict(best_state)
        test_metrics = eval_epoch(model, test_loader, criterion, device, num_classes=num_classes)
        per_seed = {
            "seed": seed,
            "val_f1": best_val_f1,
            "test_acc": test_metrics["acc"],
            "test_f1": test_metrics["macro_f1"],
            "test_per_class_f1": test_metrics["per_class_f1"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "train_time_sec": train_time_sec,
        }
        seed_results.append(per_seed)
        if verbose:
            print(f"  seed={seed}  best_val_f1={best_val_f1:.4f}  test_acc={test_metrics['acc']:.4f}  test_f1={test_metrics['macro_f1']:.4f}")

        if results_dir is not None:
            text_tag = "on" if use_text else "off"
            stem = f"{model_name}_{dataset_name}_text-{text_tag}_readout-{readout}_seed{seed}"
            torch.save(best_state, os.path.join(results_dir, f"{stem}.pt"))
            payload = {
                "model": model_name,
                "dataset": dataset_name,
                "use_text": use_text,
                "readout": readout,
                "seed": seed,
                "split_seed": 0,
                "val_f1": best_val_f1,
                "test_f1": test_metrics["macro_f1"],
                "test_accuracy": test_metrics["acc"],
                "test_per_class_f1": test_metrics["per_class_f1"],
                "confusion_matrix": test_metrics["confusion_matrix"],
                "hyperparams": {
                    "model_kwargs": {k: v for k, v in model_kwargs.items() if not isinstance(v, torch.Tensor)},
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "patience": patience,
                    "warmup_ratio": warmup_ratio,
                    "lap_pe_sign_flip": lap_pe_sign_flip,
                    "max_nodes_per_batch": max_nodes_per_batch,
                },
                "train_time_sec": train_time_sec,
            }
            with open(os.path.join(results_dir, f"{stem}.json"), "w") as f:
                json.dump(payload, f, indent=2)

    test_f1s = [r["test_f1"] for r in seed_results]
    test_accs = [r["test_acc"] for r in seed_results]
    return {
        "per_seed": seed_results,
        "test_f1_mean": float(np.mean(test_f1s)),
        "test_f1_std":  float(np.std(test_f1s)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std":  float(np.std(test_accs)),
    }
