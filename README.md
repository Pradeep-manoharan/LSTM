# Sacramento-style dendritic microcircuit (Intel Lava + MNIST proof)

> **Canonical entrypoints:** `train_mnist.py`, `demo_lava_microcircuit.py`,
> `sacramento/`, `scripts/smoke_test.py`, `artifacts/`, `RESULTS.md`.
>
> This repository **replaced** an unrelated LSTM tutorial. Nothing from that
> project is kept.

Proof-of-concept for a **Sacramento et al.**-style microcircuit
([arXiv:1810.11393](https://arxiv.org/abs/1810.11393)):

1. **3 compartments** — basal (bottom-up), soma (activity), apical (PE after cancel)
2. **SST-like cancel** — local inhibition cancels self-generated top-down on apical
3. **Local PE plasticity** — \(\Delta w \approx \mathrm{PE} \times \mathrm{presynaptic}\) (not classical backprop)
4. **Locality** — optional sparse basal / apical masks; plasticity local to the unit
5. **MNIST** — real subset runs with printed metrics (no invented numbers)
6. **No Loihi silicon** — CPU simulation only

## What runs where

| Component | Backend | Role |
|-----------|---------|------|
| MNIST train/eval | NumPy **rate model** (`sacramento/rate_model.py`) | Full PE learning + metrics |
| Microcircuit structure | **lava-nc** Process (`sacramento/lava_microcircuit.py`) | 3-compartment + SST cancel demo on CPU |
| lava-dl | Optional | Installed in the 3.10 env if you want it; MNIST PE rule is not SLAYER/STDP |

Intel Lava’s built-in learning rules are **STDP / R-STDP**, not PE×pre. Full MNIST
PE learning is therefore the paper’s **simplified two-pass rate dynamics**
(same class of model as their TensorFlow/Mathematica rate experiments), with a
**real Lava Process** proving the compartment + cancel structure when `lava-nc`
imports.

If `lava-nc` is missing, `demo_lava_microcircuit.py` still runs the **same cancel
equation** in NumPy and labels `backend: numpy-reference`. Do not treat that as
a Lava Process run.

## Environment (required for Intel Lava)

`lava-nc==0.10.0` needs **Python ≥3.10,<3.11**.

- **Python 3.13 cannot install `lava-nc`** (`Requires-Python >=3.10,<3.11`).
- **`pip install lava` on 3.13 installs an unrelated Vulkan package** (`lava==0.4.1`), not Intel Lava.
- The rate trainer (`train_mnist.py`, `scripts/smoke_test.py`) runs on 3.10–3.12 with `requirements-rate.txt`.

```bash
# Intel Lava path (CPython 3.10) — conda
conda create -n lava310 python=3.10
conda activate lava310
pip install -r requirements.txt

# Intel Lava path — uv (used on the Cloud Agent for the Lava Process run)
uv python install 3.10
uv venv --python 3.10 .venv-lava310
uv pip install --python .venv-lava310 lava-nc==0.10.0 numpy==1.26.4
PYTHONPATH=. .venv-lava310/bin/python demo_lava_microcircuit.py

# Rate-only path (Python 3.10–3.12 — no lava-nc wheel on 3.12+)
python3 -m pip install -r requirements-rate.txt
```

Versions used on the **historical local proof** (see RESULTS.md §2):

- Python 3.10.21
- lava-nc 0.10.0
- lava-dl 0.6.0
- numpy 1.26.4
- torch 2.3.1 / torchvision 0.18.1

## Commands

```bash
# Cancel demo (Lava Process on 3.10; NumPy fallback otherwise)
python demo_lava_microcircuit.py

# PE×pre smoke + cancel (no MNIST download)
python scripts/smoke_test.py

# MNIST (rate microcircuit; subset OK)
python train_mnist.py \
  --train-size 2000 --test-size 500 --epochs 20 --hidden 256 \
  --batch-size 10 --eta-pp 0.05 --weight-scale 4.0 \
  --lambda-hidden 0.35 --basal-sparsity 0.7 --apical-sparsity 0.7 \
  --eta-pi 0.0 --eta-ip 0.0 --seed 0
```

Artifacts land in `artifacts/` (`mnist_run.json`, `lava_cancel_demo.json`).

## Encoding & network

- **Input**: MNIST pixels in \([0,1]\) as rates (no spike encoding in the rate path).
- **Transfer**: \(\phi(u)=\sigma(u)\) logistic.
- **Targets**: 1-hot rates \(\{0.1, 0.8\}\) (paper-style), nudged into output soma.
- **Default reported architecture**: 784 → 256 → 10, basal/apical keep-prob 0.7.
- **Dynamics**: paper’s two-pass simplified MNIST relaxation (bottom-up, nudge output,
  reverse SST/apical/soma, then local PE updates).
- **SST**: one SST unit per next-layer pyramidal (paper 1:1). `W_PI` initialized
  as \(-\,W_{fb}\) (self-predicting cancel). Main MNIST run freezes PI/IP
  (\(\eta=0\)).

## Layout

```
.
  sacramento/rate_model.py            # PE learner (MNIST)
  sacramento/lava_microcircuit.py     # demo runner + NumPy fallback
  sacramento/lava_proc/pyramidal.py   # Lava Process + ProcessModel (module-level)
  train_mnist.py
  demo_lava_microcircuit.py
  scripts/smoke_test.py
  RESULTS.md                          # this-checkout metrics + historical
  README.md
  requirements.txt                    # lava-nc + torch (py3.10)
  requirements-rate.txt               # numpy/torch only
  artifacts/                          # mnist_run.json, lava_cancel_demo.json
```

See **RESULTS.md** for measured numbers, PE verdict, and gaps vs the original paper.
