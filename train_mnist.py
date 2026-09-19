#!/usr/bin/env python3
"""
Train Sacramento-style rate microcircuit on MNIST (tiny subset by default).

Usage:
  python train_mnist.py --train-size 500 --test-size 200 --epochs 5 --hidden 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sacramento.rate_model import SacramentoConfig, SacramentoRateNet, pe_learning_smoke


def load_mnist_subset(train_size: int, test_size: int, data_dir: Path, seed: int = 0):
    from torchvision import datasets, transforms

    data_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)

    rng = np.random.default_rng(seed)

    def _subset(ds, n):
        n = min(n, len(ds))
        idx = rng.choice(len(ds), size=n, replace=False)
        xs, ys = [], []
        for i in idx:
            img, lab = ds[int(i)]
            xs.append(img.numpy().reshape(-1))
            ys.append(lab)
        x = np.stack(xs).astype(np.float64)
        y = np.asarray(ys, dtype=np.int64)
        return x, y

    x_train, y_train = _subset(train_ds, train_size)
    x_test, y_test = _subset(test_ds, test_size)
    return x_train, y_train, x_test, y_test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-size", type=int, default=500)
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--eta-pp", type=float, default=0.05)
    p.add_argument("--basal-sparsity", type=float, default=1.0)
    p.add_argument("--apical-sparsity", type=float, default=1.0)
    p.add_argument("--weight-scale", type=float, default=3.0)
    p.add_argument("--eta-pi", type=float, default=0.0, help="0 keeps SST->apical cancel fixed (paper MNIST)")
    p.add_argument("--eta-ip", type=float, default=0.0)
    p.add_argument("--lambda-hidden", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=str(ROOT / "artifacts" / "mnist_run.json"))
    args = p.parse_args()

    print("=== Sacramento MNIST (rate microcircuit) ===")
    print(f"Python: {sys.version}")
    try:
        import importlib.metadata as md

        import lava  # noqa: F401

        print(f"lava-nc: {md.version('lava-nc')}")
        try:
            print(f"lava-dl: {md.version('lava-dl')}")
        except Exception:
            print("lava-dl: not installed")
    except Exception as e:
        print(f"Lava import note: {e}")

    print("--- PE learning smoke test ---")
    smoke = pe_learning_smoke(seed=args.seed + 1)
    print(json.dumps(smoke, indent=2))

    data_dir = ROOT / "artifacts" / "mnist"
    print(f"--- Loading MNIST subset train={args.train_size} test={args.test_size} ---")
    x_tr, y_tr, x_te, y_te = load_mnist_subset(
        args.train_size, args.test_size, data_dir, seed=args.seed
    )
    print(f"x_train {x_tr.shape} y_train {y_tr.shape} class counts {np.bincount(y_tr, minlength=10)}")
    print(f"x_test  {x_te.shape} y_test  {y_te.shape}")

    cfg = SacramentoConfig(
        layer_sizes=(784, args.hidden, 10),
        eta_pp=args.eta_pp,
        eta_ip=args.eta_ip,
        eta_pi=args.eta_pi,
        eta_bias=args.eta_pp,
        lambda_hidden=args.lambda_hidden,
        lambda_out=0.5,
        lambda_sst=0.5,
        basal_sparsity=args.basal_sparsity,
        apical_sparsity=args.apical_sparsity,
        seed=args.seed,
        weight_scale=args.weight_scale,
    )
    net = SacramentoRateNet(cfg)
    print(
        f"Network: {cfg.layer_sizes} basal_sparse={cfg.basal_sparsity} "
        f"apical_sparse={cfg.apical_sparsity}"
    )
    n_pp = int(np.sum(net.basal_mask[0] > 0))
    n_fb = int(np.sum(net.apical_mask[0] > 0)) if net.apical_mask[0] is not None else 0
    print(f"Nonzero basal W_pp[0]: {n_pp}/{net.W_pp[0].size}; apical W_fb[0]: {n_fb}")

    t0 = time.time()
    history = []
    test0 = net.evaluate(x_te, y_te)
    print(f"Epoch 0 (before train) test_acc={test0:.4f}")
    history.append({"epoch": 0, "test_acc": test0, "train_acc": None, "mean_abs_pe": None})

    for ep in range(1, args.epochs + 1):
        tr = net.train_epoch(x_tr, y_tr, batch_size=args.batch_size)
        te = net.evaluate(x_te, y_te)
        row = {
            "epoch": ep,
            "train_acc": tr["train_acc"],
            "test_acc": te,
            "mean_abs_pe": tr["mean_abs_pe"],
            "mean_abs_dw": tr["mean_abs_dw_pp"],
        }
        history.append(row)
        print(
            f"Epoch {ep}/{args.epochs} train_acc={tr['train_acc']:.4f} "
            f"test_acc={te:.4f} mean_|PE|={tr['mean_abs_pe']:.6f} "
            f"mean_|dW|={tr['mean_abs_dw_pp']:.6e}"
        )

    elapsed = time.time() - t0
    pe_ok = smoke["weight_delta_l1"] > 0 and smoke["pe_before"] >= 0
    result = {
        "config": {
            "layer_sizes": list(cfg.layer_sizes),
            "train_size": args.train_size,
            "test_size": args.test_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "basal_sparsity": args.basal_sparsity,
            "apical_sparsity": args.apical_sparsity,
            "eta_pp": args.eta_pp,
            "eta_pi": args.eta_pi,
            "eta_ip": args.eta_ip,
            "weight_scale": args.weight_scale,
            "lambda_hidden": args.lambda_hidden,
            "seed": args.seed,
        },
        "smoke": smoke,
        "history": history,
        "final_train_acc": history[-1]["train_acc"],
        "final_test_acc": history[-1]["test_acc"],
        "elapsed_sec": elapsed,
        "pe_learning_verdict": (
            "PASS: local PE×pre updated basal weights "
            f"(Δ|W|={smoke['weight_delta_l1']:.6e}); "
            "MNIST run used same rule (Eq. 7–9 style)."
            if pe_ok
            else "FAIL: no weight change in smoke test"
        ),
        "python": sys.version,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"Elapsed {elapsed:.1f}s")
    print("PE verdict:", result["pe_learning_verdict"])


if __name__ == "__main__":
    main()
