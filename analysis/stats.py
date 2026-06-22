"""Aggregation and paired t-tests over the results/ directory layout.

Layout expected:
    results/text_off/{model}_{dataset}.json
    results/text_on/{model}_{dataset}.json

Each JSON has the shape written by run_experiment:
    {
      "per_seed": [{"seed": N, "val_f1": ..., "test_f1": ..., "test_acc": ..., ...}, ...],
      "test_f1_mean": ..., "test_f1_std": ..., "test_acc_mean": ..., "test_acc_std": ...
    }

Model name encodes readout for GPS: "gps_4way" vs "gps_mean". All others have no readout suffix.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats


def load_results(results_dir: str) -> dict[tuple[str, str, bool], dict]:
    """Return a dict keyed by (model, dataset, use_text) → aggregate JSON dict.

    Scans results/text_off/ and results/text_on/ under results_dir.
    Filename: {model}_{dataset}.json  (e.g. gps_4way_twitter15.json)
    """
    out: dict[tuple[str, str, bool], dict] = {}
    for use_text, subdir in ((False, "text_off"), (True, "text_on")):
        pattern = os.path.join(results_dir, subdir, "*.json")
        for path in sorted(glob.glob(pattern)):
            stem = Path(path).stem  # e.g. "gps_4way_twitter15"
            # dataset is always the last token (twitter15 / twitter16)
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            model, dataset = parts[0], parts[1]
            with open(path) as f:
                out[(model, dataset, use_text)] = json.load(f)
    return out


def per_seed_f1(entry: dict, metric: str = "test_f1") -> list[float]:
    return [s[metric] for s in entry["per_seed"]]


def summarise(entry: dict, metric: str = "test_f1") -> dict:
    vals = per_seed_f1(entry, metric)
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_seed": vals, "n": len(vals)}


def paired_t(a: dict, b: dict, metric: str = "test_f1") -> dict:
    """Paired t-test on per-seed metric, matching by seed index (not seed value)."""
    va = per_seed_f1(a, metric)
    vb = per_seed_f1(b, metric)
    n = min(len(va), len(vb))
    if n < 2:
        return {"n_pairs": n, "mean_delta": float("nan"), "std_delta": float("nan"),
                "t": float("nan"), "p": float("nan")}
    va, vb = np.array(va[:n]), np.array(vb[:n])
    delta = va - vb
    t_stat, p_val = stats.ttest_rel(va, vb)
    return {
        "n_pairs": n,
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std(ddof=1)),
        "t": float(t_stat),
        "p": float(p_val),
    }
