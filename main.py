"""Leave-one-subject-out fold runner -- plain PyTorch, no extra framework.

One held-out subject per run: train on the other 12, validate on a 10% split,
then evaluate every held-out recording and print the paper's metrics table.

Reuses loader.py / networks.py / metrics.py unchanged. The model is pulled from
networks.py by its registered name (torchvision's get_model). Nothing else.

Usage (any env with torch + torchvision + scipy + scikit-learn):
  python main.py --patient 0003 --seed 42

All patients: 0001 0002 0003 0005 0006 0007 1000 1001 1002 1003 1004 1101 1102
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import get_model

import loader
import networks  # noqa: F401  -- side-effect: registers the models with torchvision
from metrics import leaky_accuracy

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train")

PATIENTS = ["0001", "0002", "0003", "0005", "0006", "0007",
            "1000", "1001", "1002", "1003", "1004", "1101", "1102"]
THRESHOLDS = [0.05, 0.10, 0.20]


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def features(batch: dict) -> dict:
    """Model inputs = every batch key except the target."""
    return {k: v for k, v in batch.items() if k != "observation"}


def criterion(task: str):
    if task == "classification":
        return lambda p, t: F.binary_cross_entropy(p.flatten(), t.flatten())
    return lambda p, t: F.l1_loss(p.flatten(), t.flatten())


def validate(model, loader_, loss_fn, device) -> float:
    """Mean loss over a loader (no grad), weighted by sample count."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader_:
            pred = model(**features(batch))
            targ = batch["observation"].to(device)
            total += loss_fn(pred, targ).item() * len(targ)
            n += len(targ)
    model.train()
    return total / max(n, 1)


def train(model, train_ds, valid_ds, args, device):
    """Train with ReduceLROnPlateau + early stopping; restore best weights."""
    loss_fn = criterion(args.task)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)

    val_every, patience, delta = 10, 15, 1e-4
    best_val, best_state, stale = float("inf"), None, 0
    last_train, last_val, epochs_run = 0.0, None, 0

    model.train()
    for epoch in range(args.num_epochs):
        epochs_run = epoch + 1
        running, nb = 0.0, 0
        for batch in train_loader:
            pred = model(**features(batch))
            loss = loss_fn(pred, batch["observation"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            nb += 1
        last_train = running / max(nb, 1)

        is_val = ((epoch + 1) % val_every == 0) or (epoch == args.num_epochs - 1)
        if is_val:
            last_val = validate(model, valid_loader, loss_fn, device)
            scheduler.step(last_val)
            if last_val < best_val - delta:
                best_val, stale = last_val, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    logger.info("early stop at epoch %d (best val=%.6f)", epoch + 1, best_val)
                    break

        if (epoch + 1) % 30 == 0 or epoch == 0:
            vs = f" val={last_val:.6f}" if last_val is not None else ""
            logger.info("epoch %d/%d  train=%.6f%s  lr=%.2e",
                        epoch + 1, args.num_epochs, last_train, vs,
                        optimizer.param_groups[0]["lr"])

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info("restored best weights (val=%.6f)", best_val)
    return last_train, best_val if best_state is not None else last_val, epochs_run


def evaluate(model, test_dicts, train_mean):
    """Per-recording MAE / MSE / leaky-acc + mean-predictor baseline."""
    model.eval()
    rows = []
    for name, rec_ds in test_dicts.items():
        if len(rec_ds) == 0:
            continue
        with torch.no_grad():
            batch = next(iter(DataLoader(rec_ds, batch_size=len(rec_ds), shuffle=False)))
            preds = model(**features(batch)).flatten().cpu().numpy()
            targs = batch["observation"].flatten().cpu().numpy()

        means = np.full_like(preds, train_mean)
        row = {"recording": name, "n": int(len(targs)),
               "mae": float(np.abs(preds - targs).mean()),
               "mse": float(((preds - targs) ** 2).mean()),
               "mean_mae": float(np.abs(means - targs).mean()),
               "mean_mse": float(((means - targs) ** 2).mean())}
        if np.any(targs > 0.0):
            for eps in THRESHOLDS:
                acc, _, _, _ = leaky_accuracy(preds, targs, span=5, threshold=eps)
                macc, _, _, _ = leaky_accuracy(means, targs, span=5, threshold=eps)
                row[f"leaky_acc_{eps:.2f}"] = float(acc)
                row[f"mean_leaky_acc_{eps:.2f}"] = float(macc)
        rows.append(row)
    return rows


def print_table(patient, rows):
    print(f"\n{'='*70}\nPatient {patient} -- Test Results\n{'='*70}")
    header = f"{'Recording':<12} {'MAE':>8} {'MSE':>8} {'uMAE':>8} {'uMSE':>8}"
    for eps in THRESHOLDS:
        header += f" {'LA@'+str(eps):>8}"
    print(header + "\n" + "-" * 70)
    for r in rows:
        line = (f"{r['recording']:<12} {r['mae']:>8.4f} {r['mse']:>8.4f} "
                f"{r['mean_mae']:>8.4f} {r['mean_mse']:>8.4f}")
        for eps in THRESHOLDS:
            line += f" {r.get(f'leaky_acc_{eps:.2f}', float('nan')):>8.4f}"
        print(line)
    print("=" * 70 + "\n")


def main():
    p = argparse.ArgumentParser(description="EEG cybersickness leave-one-subject-out fold")
    p.add_argument("--patient", required=True, help="4-char patient ID, e.g. 0003")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--input-type", default="power-spectral-difference",
                   choices=["power-spectral-difference", "power-spectral-no-kinematic",
                            "power-spectral-no-eeg", "kinematic"])
    p.add_argument("--task", default="regression", choices=["regression", "classification"])
    p.add_argument("--cache-prefix", default="/home/adhd/src/research/datasets/juliete/.cache")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--smoke", action="store_true", help="run only 3 epochs")
    p.add_argument("--out", default=None, help="append a JSON line of fold results here")
    args = p.parse_args()

    args.patient = f"{int(args.patient):04d}"
    if args.smoke:
        args.num_epochs = 3
    set_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda else "cuda")

    logger.info("fold patient=%s seed=%d epochs=%d input=%s",
                args.patient, args.seed, args.num_epochs, args.input_type)

    train_ds, valid_ds, test_dicts = loader.load_train_test_datasets(
        prefix=args.cache_prefix, patient=args.patient,
        input_type=args.input_type, task=args.task, validation=True)
    train_mean = float(np.mean(train_ds.train_mean)) if train_ds.train_mean is not None else 0.0

    hidden_size = 32 if args.input_type == "kinematic" else 64
    model = get_model(f"{args.input_type}-model",
                      n_channels=16, hidden_size=hidden_size, num_classes=1).to(device)

    last_train, val_loss, epochs_run = train(model, train_ds, valid_ds, args, device)
    rows = evaluate(model, test_dicts, train_mean)
    print_table(args.patient, rows)

    if args.out:
        record = {"patient": args.patient, "seed": args.seed,
                  "input_type": args.input_type, "task": args.task,
                  "epochs_run": epochs_run, "status": "completed",
                  "train_loss": last_train, "val_loss": val_loss, "rows": rows}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.info("appended fold results to %s", args.out)


if __name__ == "__main__":
    main()
