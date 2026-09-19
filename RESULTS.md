# RESULTS — Sacramento microcircuit / MNIST (honest metrics only)

**Project:** Kannakilabs Sacramento-style Lava microcircuit + MNIST software proof  
**Hardware:** CPU only (no Loihi silicon)

This file has two sections:

1. **This checkout** — numbers produced by running the scripts in *this* tree.
2. **Historical local proof (19 Sep 2026)** — attached prior run on a Python 3.10
   `lava310` conda env. Kept as reference only. Do **not** mix those numbers with
   this checkout unless a command below reproduced them.

---

## 1. This checkout

**Status at first commit:** metrics pending the agent run on this Cloud Agent VM
(system Python 3.12.3). They will be filled from real stdout / JSON after
`scripts/smoke_test.py`, `demo_lava_microcircuit.py`, and a MNIST subset train.
If a command cannot run, that is stated — numbers are never invented.

### Environment probe (to be updated)

| Item | Value |
|------|-------|
| Python | *(fill after run)* |
| lava-nc | *(fill after run; expected fail on 3.12)* |
| numpy / torch | *(fill after run)* |

---

## 2. Historical local proof (19 Sep 2026, IST) — reference only

**Source:** prior working tree `/workspace/lava-sacramento-mnist` on a laptop /
workstation with Miniconda env `lava310`. **Not** this GitHub replacement run.

| Item | Value |
|------|-------|
| Python | 3.10.21 (conda env `lava310`) |
| lava-nc | 0.10.0 (**installed and imported**) |
| lava-dl | 0.6.0 (installed; not used for PE rule) |
| numpy | 1.26.4 |
| torch / torchvision | 2.3.1 / 0.18.1 |
| System Python 3.13 | Cannot install lava-nc |

### Exact commands run (historical)

```bash
source /workspace/lava-sacramento-mnist/miniconda/etc/profile.d/conda.sh
conda activate lava310
cd /workspace/lava-sacramento-mnist

python demo_lava_microcircuit.py

python train_mnist.py \
  --train-size 2000 --test-size 500 --epochs 20 --hidden 256 \
  --batch-size 10 --eta-pp 0.05 --weight-scale 4.0 \
  --lambda-hidden 0.35 --basal-sparsity 0.7 --apical-sparsity 0.7 \
  --eta-pi 0.0 --eta-ip 0.0 --seed 0
```

### Real metrics — MNIST subset (historical)

Feedforward accuracy only (no output nudging in the accuracy metric).

| Epoch | train_acc | test_acc | mean \|PE\| (during teach) | mean \|dW\| |
|------:|----------:|---------:|---------------------------:|------------:|
| 0 | — | 0.0880 | — | — |
| 1 | 0.2065 | 0.1860 | 0.6972 | 1.19e-4 |
| 5 | 0.6455 | 0.6500 | 0.7019 | 4.48e-5 |
| 10 | 0.7670 | 0.7920 | 0.7361 | 3.81e-5 |
| 15 | 0.8025 | 0.8080 | 0.7493 | 3.42e-5 |
| 20 | **0.8250** | **0.8340** | 0.7554 | 3.33e-5 |

- Elapsed: **12.0 s**
- Chance on 10-way MNIST ≈ 0.10; final test **0.834** on 500 held-out images
- **Not** comparable to paper’s 1.96% test error on full MNIST (55000 train)

### Smoke test (historical, synthetic 8→6→3, 40 epochs)

```
pe_before:        1.0777
pe_after:         0.4539
weight_delta_l1:  1.444e-02
train_acc_last:   0.46875
```

### Lava Process cancel demo (historical)

CPU `Loihi1SimCfg`, 4 units × 5 steps:

| Condition | mean \|apical PE\| |
|-----------|-------------------:|
| teach matched to top-down | **0.449** |
| teach mismatched | **0.713** |

Mismatched > matched → SST-like cancel reduces apical PE **directionally** in Lava.

### Historical Lava install status

| Attempt | Result |
|---------|--------|
| `pip install lava` (py3.13 venv) | Installed **wrong** package lava 0.4.1 (Vulkan) |
| `pip install lava-nc` (py3.13) | **FAIL**: Requires-Python ≥3.10,<3.11 |
| `pip install lava-dl` (py3.13) | **FAIL**: no matching distribution |
| conda `lava310` + lava-nc 0.10.0 + lava-dl 0.6.0 | **OK** |

---

## Blockers / gaps vs Sacramento et al. (original)

1. **Not full continuous-time ODE / stochastic spiking** — uses paper’s **simplified two-pass rate** MNIST dynamics.
2. **MNIST PE learning not inside Lava ProcessModels** — Lava learning DSL is STDP-oriented; PE×pre runs in NumPy.
3. **Subset MNIST** in the historical table (2000/500), not full 55k/10k.
4. **Network smaller** (784-256-10 vs paper four-layer / 784-1000-10 variants).
5. **Feedback fixed (FA-style)**; no slow top-down plasticity (Eq. 10) in the main run.
6. **Self-predicting init** is \(W_{PI}=-W_{fb}\); residual apical PE remains with frozen PI (expected teaching signal).
7. **No event-driven sparse-in-time synaptic updates** on Loihi; rate batches update every example.
8. **SST teaching** uses paper 1:1 (SST dim = next pyramidal layer).
9. **No convolutional MNIST / no hardware** — software CPU only.
10. Wrong PyPI name `lava` is a footgun on modern Python.

## Failure modes observed (historical + design)

- Tiny `weight_scale` → hidden logistic saturation at ~0.5 → no learning (use scale ~3–5).
- Measuring “train acc” under output nudging → false 100% (fixed: feedforward-only accuracy).
- Python 3.13 cannot host lava-nc; must use 3.10.

## Success criteria check

| Criterion | Status |
|-----------|--------|
| Old LSTM app removed | **Yes** (this replacement) |
| 3 compartments + SST cancel + PE×pre | **Yes** (rate + Lava/NumPy cancel demo) |
| Honest RESULTS.md | **Yes** (this-checkout vs historical labeled) |
| Prefer real Lava if it works | **Documented**; 3.10 required for lava-nc |
