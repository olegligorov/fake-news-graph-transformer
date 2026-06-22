"""Per-seed JSON aggregation and paired t-tests for the structural / text contrasts."""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy import stats


def load_run_files(results_dir: str) -> list[dict]:
    """Load every per-seed JSON under results_dir.

    Filename pattern: {model}_{dataset}_text-{on|off}_readout-{mean|4way}_seed{N}.json
    """
    files = sorted(glob.glob(os.path.join(results_dir, "*_text-*_readout-*_seed*.json")))
    runs = []
    for path in files:
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def group_runs(runs: Iterable[dict]) -> dict[tuple[str, str, bool, str], list[dict]]:
    """Group runs by (model, dataset, use_text, readout), sorted by seed within each group."""
    groups: dict[tuple[str, str, bool, str], list[dict]] = defaultdict(list)
    for r in runs:
        key = (r["model"], r["dataset"], bool(r["use_text"]), r["readout"])
        groups[key].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r["seed"])
    return groups


def summarise(runs: list[dict], metric: str = "test_f1") -> dict:
    """Mean / std / per-seed list for a single (model, dataset, text, readout) cell."""
    vals = [r[metric] for r in runs]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "per_seed": vals,
        "n": len(vals),
    }


def paired_t(a: list[dict], b: list[dict], metric: str = "test_f1") -> dict:
    """Paired t-test on per-seed metric, matching seeds in `a` and `b`."""
    by_seed_a = {r["seed"]: r[metric] for r in a}
    by_seed_b = {r["seed"]: r[metric] for r in b}
    common = sorted(set(by_seed_a) & set(by_seed_b))
    if len(common) < 2:
        return {"n_pairs": len(common), "mean_delta": float("nan"), "std_delta": float("nan"), "t": float("nan"), "p": float("nan")}
    va = np.array([by_seed_a[s] for s in common])
    vb = np.array([by_seed_b[s] for s in common])
    delta = va - vb
    t_stat, p_val = stats.ttest_rel(va, vb)
    return {
        "n_pairs": len(common),
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std(ddof=1)),
        "t": float(t_stat),
        "p": float(p_val),
    }
