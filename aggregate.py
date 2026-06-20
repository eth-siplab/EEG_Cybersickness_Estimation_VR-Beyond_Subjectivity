"""Aggregate leave-one-out fold results into the paper's Table metrics.

Reads one or more results .jsonl files (from main.py --out) and reports,
per input-type, the pooled MAE / MSE / leaky-accuracy across all held-out test
recordings, alongside the mean-predictor baseline.  Macro = unweighted mean over
recordings; micro = segment-count-weighted.

Usage:
  python aggregate.py results_power-spectral-difference.jsonl [more.jsonl ...]
"""
import sys
import json
import numpy as np


def load(paths):
    recs = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def agg(folds):
    rows, seeds, patients = [], set(), set()
    for f in folds:
        seeds.add(f["seed"])
        patients.add(f["patient"])
        rows.extend(f["rows"])
    n = np.array([r["n"] for r in rows], dtype=float)

    def wmean(key):
        v = np.array([r[key] for r in rows], dtype=float)
        m = np.isfinite(v)
        macro = float(np.nanmean(v[m])) if m.any() else float("nan")
        micro = float(np.sum(v[m] * n[m]) / np.sum(n[m])) if m.any() else float("nan")
        return macro, micro

    out = {"n_folds": len(folds), "n_recordings": len(rows),
           "patients": len(patients), "seeds": sorted(seeds)}
    for key in ["mae", "mse", "mean_mae", "mean_mse"]:
        out[key] = wmean(key)
    for eps in [0.05, 0.10, 0.20]:
        for pref in ["leaky_acc", "mean_leaky_acc"]:
            k = f"{pref}_{eps:.2f}"
            present = [r for r in rows if k in r and np.isfinite(r[k])]
            out[k] = (float(np.mean([r[k] for r in present])) if present else float("nan"),
                      len(present))
    return out


def across_seeds(fs):
    """Paper protocol: pool per seed, then mean +/- std across seeds."""
    by_seed = {}
    for f in fs:
        by_seed.setdefault(f["seed"], []).append(f)
    keys = ["mae", "mse", "mean_mae", "mean_mse",
            "leaky_acc_0.10", "mean_leaky_acc_0.10"]
    per_seed = {k: [] for k in keys}
    for seed, sf in by_seed.items():
        a = agg(sf)
        for k in keys:
            v = a[k][0] if isinstance(a[k], tuple) else a[k]
            if np.isfinite(v):
                per_seed[k].append(v)
    return {k: (float(np.mean(v)) if v else float("nan"),
               float(np.std(v)) if len(v) > 1 else 0.0) for k, v in per_seed.items()}, sorted(by_seed)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    folds = load(sys.argv[1:])
    by_type = {}
    for f in folds:
        by_type.setdefault(f["input_type"], []).append(f)

    order = ["kinematic", "power-spectral-no-eeg",
             "power-spectral-no-kinematic", "power-spectral-difference"]
    for it in sorted(by_type, key=lambda x: order.index(x) if x in order else 99):
        fs = by_type[it]
        a = agg(fs)
        s, seeds = across_seeds(fs)
        multi = len(seeds) > 1
        print(f"\n{'='*64}\ninput_type = {it}")
        print(f"  folds={a['n_folds']}  patients={a['patients']}  seeds={seeds}  test_recordings={a['n_recordings']}")
        if multi:
            print(f"  (mean +/- std across {len(seeds)} seeds; model vs mean-baseline)")
            print(f"  MAE            {s['mae'][0]:.4f} +/- {s['mae'][1]:.4f}    baseline {s['mean_mae'][0]:.4f}")
            print(f"  MSE            {s['mse'][0]:.4f} +/- {s['mse'][1]:.4f}    baseline {s['mean_mse'][0]:.4f}")
            print(f"  acc@0.10       {s['leaky_acc_0.10'][0]*100:.2f} +/- {s['leaky_acc_0.10'][1]*100:.2f} %  baseline {s['mean_leaky_acc_0.10'][0]*100:.2f} %")
        else:
            print(f"  {'metric':<22}{'model (macro/micro)':<26}{'mean-baseline'}")
            print(f"  {'MAE':<22}{a['mae'][0]:.4f} / {a['mae'][1]:.4f}{'':<8}{a['mean_mae'][0]:.4f} / {a['mean_mae'][1]:.4f}")
            print(f"  {'MSE':<22}{a['mse'][0]:.4f} / {a['mse'][1]:.4f}{'':<8}{a['mean_mse'][0]:.4f} / {a['mean_mse'][1]:.4f}")
            for eps in [0.05, 0.10, 0.20]:
                mk, mn = a[f"leaky_acc_{eps:.2f}"]
                bk, _ = a[f"mean_leaky_acc_{eps:.2f}"]
                print(f"  leaky-acc@{eps:.2f} (n={mn:<3}) {mk*100:6.2f}%{'':<13}{bk*100:6.2f}%")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
