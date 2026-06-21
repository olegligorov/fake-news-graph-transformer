import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


def stratified_split(dataset: Dataset, train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 42):
    """Return (train_idx, val_idx, test_idx) lists with stratification on y."""
    labels = [dataset[i].y.item() for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    test_ratio = 1.0 - train_ratio - val_ratio

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=(1.0 - train_ratio), stratify=labels, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_ratio / (test_ratio + val_ratio), stratify=temp_labels, random_state=seed
    )
    return train_idx, val_idx, test_idx


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        loss = criterion(out, batch.y.view(-1))
        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y.view(-1).cpu().tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": acc,
        "macro_f1": macro_f1,
    }


def run_experiment(
    model_cls,
    model_kwargs: dict,
    dataset: Dataset,
    seeds: list[int],
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Train model_cls for each seed; return per-seed and aggregate results.

    Args:
        patience: epochs without val_f1 improvement before stopping. None = disabled.
    """
    criterion = nn.CrossEntropyLoss()
    seed_results = []

    for seed in seeds:
        _set_seed(seed)
        train_idx, val_idx, test_idx = stratified_split(dataset, seed=seed)

        train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader([dataset[i] for i in val_idx],   batch_size=batch_size)
        test_loader  = DataLoader([dataset[i] for i in test_idx],  batch_size=batch_size)

        model = model_cls(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best_val_f1 = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = eval_epoch(model, val_loader, criterion, device)
            scheduler.step(val_metrics["loss"])

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and epoch % 20 == 0:
                print(f"  seed={seed} epoch={epoch:3d}  train_loss={train_loss:.4f}  val_f1={val_metrics['macro_f1']:.4f}  best={best_val_f1:.4f}")

            if patience is not None and epochs_no_improve >= patience:
                if verbose:
                    print(f"  seed={seed} early stop at epoch {epoch}")
                break

        model.load_state_dict(best_state)
        test_metrics = eval_epoch(model, test_loader, criterion, device)
        seed_results.append({
            "seed": seed,
            "val_f1": best_val_f1,
            "test_acc": test_metrics["acc"],
            "test_f1": test_metrics["macro_f1"],
        })
        if verbose:
            print(f"  seed={seed}  best_val_f1={best_val_f1:.4f}  test_acc={test_metrics['acc']:.4f}  test_f1={test_metrics['macro_f1']:.4f}")

    test_f1s = [r["test_f1"] for r in seed_results]
    test_accs = [r["test_acc"] for r in seed_results]
    return {
        "per_seed": seed_results,
        "test_f1_mean": float(np.mean(test_f1s)),
        "test_f1_std":  float(np.std(test_f1s)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std":  float(np.std(test_accs)),
    }
