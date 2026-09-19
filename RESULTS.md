# RESULTS — Sacramento microcircuit / MNIST (honest metrics only)

**Project:** Kannakilabs Sacramento-style Lava microcircuit + MNIST software proof  
**Hardware:** CPU only (no Loihi silicon)

This file has two sections:

1. **This checkout** — numbers printed by the scripts in *this* tree on the Cloud Agent VM (19 Sep 2026, UTC).
2. **Historical local proof (19 Sep 2026, IST)** — attached prior run on a Python 3.10 `lava310` conda env. **Reference only.** Do not mix those numbers with section 1.

JSON for section 1: `artifacts/mnist_run.json`, `artifacts/lava_cancel_demo.json`.

---

## 1. This checkout (Cloud Agent, 19 Sep 2026 UTC)

### Environment

| Item | Value |
|------|-------|
| Rate trainer / MNIST | CPython **3.12.3** (system) |
| numpy | 1.26.4 |
| torch / torchvision | 2.14.0+cu130 / 0.29.0+cu130 (MNIST download only) |
| lava-nc on 3.12 | **Not importable** (`No module named 'lava'`) — no 3.12 wheel |
| Lava Process demo | CPython **3.10.21** (`uv` managed) + **lava-nc 0.10.0** |
| lava-dl | Not installed in this run (not used for PE) |

`pip install lava-nc` was **not** attempted on 3.12 after the known `<3.11` constraint. A separate 3.10 venv was created with `uv python install 3.10` and `uv pip install lava-nc==0.10.0`.

### Exact commands

```bash
# Rate path (system 3.12)
python3 -m pip install -r requirements-rate.txt
python3 scripts/smoke_test.py
python3 demo_lava_microcircuit.py          # backend: numpy-reference
python3 train_mnist.py \
  --train-size 2000 --test-size 500 --epochs 20 --hidden 256 \
  --batch-size 10 --eta-pp 0.05 --weight-scale 4.0 \
  --lambda-hidden 0.35 --basal-sparsity 0.7 --apical-sparsity 0.7 \
  --eta-pi 0.0 --eta-ip 0.0 --seed 0

# Lava Process (uv CPython 3.10.21 + lava-nc 0.10.0)
PYTHONPATH=. /tmp/lava310/bin/python demo_lava_microcircuit.py
PYTHONPATH=. /tmp/lava310/bin/python scripts/smoke_test.py
```

### PE learning smoke (synthetic 8→6→3, 40 epochs, seed=1)

Same numbers on 3.12 and 3.10 (NumPy):

| Field | Value |
|-------|------:|
| pe_before | 0.1372 |
| pe_after | 0.1038 |
| weight_delta_l1 | 13.06 |
| train_acc_last | 0.5000 |

Weights moved under \(\Delta w \propto \mathrm{PE}\times r_{\mathrm{pre}}\); teaching PE decreased. Verdict: **PASS**.

### Cancel demo

Same stimuli (n=4, steps=5, seed=0). Matched SST teach vs mismatched:

| Backend | mean \|apical PE\| matched | mismatched |
|---------|---------------------------:|-----------:|
| **lava-nc 0.10.0** (`Loihi1SimCfg`, py3.10) | **0.2972** | **0.4152** |
| numpy-reference (py3.12, lava missing) | 0.2972 | 0.4152 |

Mismatched > matched → SST-like cancel reduces apical PE **directionally**. The Lava Process and the NumPy equation agree to printed precision (same RNG + same update).

### MNIST subset (rate microcircuit, this checkout)

Feedforward accuracy only (no output nudging in the accuracy metric).

Architecture **784 → 256 → 10**; basal keep-prob 0.7 (nonzero \(W_{pp}[0]\): **140220 / 200704**); apical keep-prob 0.7 (nonzero \(W_{fb}[0]\): **1768 / 2560**). Batch 10; subset **train 2000 / test 500**; \(\eta_{pp}=0.05\), \(\lambda_{hidden}=0.35\), \(\lambda_{out}=0.5\); **SST→apical and pyr→SST frozen** (\(\eta=0\)).

| Epoch | train_acc | test_acc | mean \|PE\| (during teach) | mean \|dW\| |
|------:|----------:|---------:|---------------------------:|------------:|
| 0 | — | 0.0800 | — | — |
| 1 | 0.1070 | 0.0860 | 0.0524 | 1.54e-4 |
| 5 | 0.2110 | 0.2060 | 0.0420 | 9.01e-5 |
| 10 | 0.4445 | 0.4380 | 0.0391 | 7.45e-5 |
| 15 | 0.5470 | 0.4880 | 0.0368 | 6.65e-5 |
| 17 | 0.6910 | 0.6640 | 0.0358 | 6.44e-5 |
| 20 | **0.7285** | **0.6920** | 0.0343 | 6.20e-5 |

- Elapsed: **6.0 s**
- Chance on 10-way MNIST ≈ 0.10; test rose **0.080 → 0.692** on 500 held-out images
- Nonzero `mean_|dW|` every epoch; mean \|PE\| during teach fell **0.052 → 0.034**
- Accuracy is **not monotonic** (dips at epochs 11 and 15) — reported as printed
- **Not** comparable to the paper’s 1.96% test error on full MNIST (55000 train)
- **Not** the same numbers as the historical 0.834 table (different reconstructed `sacramento/` tree; see §2)

PE learning verdict: **PASS** (weights moved; feedforward test accuracy well above chance).

---

## 2. Historical local proof (19 Sep 2026, IST) — reference only

**Source:** prior working tree `/workspace/lava-sacramento-mnist` with Miniconda env `lava310`. That `sacramento/` package was **not** attached to this rebuild; section 1 is a paper-faithful reconstruction. Treat §2 as a previous local proof, not a claim that this git tree reproduced 0.834.

| Item | Value |
|------|-------|
| Python | 3.10.21 (conda env `lava310`) |
| lava-nc | 0.10.0 (**installed and imported**) |
| lava-dl | 0.6.0 (installed; not used for PE rule) |
| numpy | 1.26.4 |
| torch / torchvision | 2.3.1 / 0.18.1 |

### Historical MNIST subset (same CLI flags as §1)

| Epoch | train_acc | test_acc | mean \|PE\| (during teach) | mean \|dW\| |
|------:|----------:|---------:|---------------------------:|------------:|
| 0 | — | 0.0880 | — | — |
| 1 | 0.2065 | 0.1860 | 0.6972 | 1.19e-4 |
| 5 | 0.6455 | 0.6500 | 0.7019 | 4.48e-5 |
| 10 | 0.7670 | 0.7920 | 0.7361 | 3.81e-5 |
| 15 | 0.8025 | 0.8080 | 0.7493 | 3.42e-5 |
| 20 | **0.8250** | **0.8340** | 0.7554 | 3.33e-5 |

- Elapsed: **12.0 s**; final test **0.834** on 500 held-out images

### Historical smoke (8→6→3, 40 epochs)

```
pe_before:        1.0777
pe_after:         0.4539
weight_delta_l1:  1.444e-02
train_acc_last:   0.46875
```

### Historical Lava Process cancel demo

CPU `Loihi1SimCfg`, 4 units × 5 steps (different Process weights than §1):

| Condition | mean \|apical PE\| |
|-----------|-------------------:|
| teach matched to top-down | **0.449** |
| teach mismatched | **0.713** |

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
3. **Subset MNIST** (2000/500), not full 55k/10k.
4. **Network smaller** (784-256-10 vs paper four-layer / 784-1000-10 variants).
5. **Feedback fixed (FA-style)**; no slow top-down plasticity (Eq. 10) in the main run.
6. **Self-predicting init** is \(W_{PI}=-W_{fb}\); residual apical PE remains with frozen PI (expected teaching signal).
7. **No event-driven sparse-in-time synaptic updates** on Loihi; rate batches update every example.
8. **SST teaching** uses paper 1:1 (SST dim = next pyramidal layer).
9. **No convolutional MNIST / no hardware** — software CPU only.
10. Wrong PyPI name `lava` is a footgun on modern Python.

## Failure modes observed

- Tiny `weight_scale` → hidden logistic saturation at ~0.5 → no learning (use scale ~3–5).
- Measuring “train acc” under output nudging → false 100% (fixed: feedforward-only accuracy).
- Python ≥3.11 cannot host lava-nc 0.10.0; use CPython 3.10.
- lava-nc compiler warns/fails if ProcessModels are nested inside functions (must be module-level). Isolated under `sacramento/lava_proc/`.

## Success criteria check

| Criterion | Status |
|-----------|--------|
| Old LSTM app removed | **Yes** |
| End-to-end run with real printed metrics | **Yes** (this checkout) |
| Honest RESULTS.md | **Yes** (§1 measured; §2 labeled historical) |
| 3 compartments + SST cancel + PE×pre | **Yes** (rate + Lava cancel demo on 3.10) |
| Prefer real Lava if it works | **lava-nc 0.10.0 ran** on CPython 3.10; PE trainer is rate (documented) |
