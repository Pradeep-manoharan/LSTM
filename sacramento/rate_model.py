"""NumPy rate implementation of Sacramento et al. (2018) simplified MNIST dynamics.

Not classical backprop. Local dendritic prediction-error plasticity:

    Δw ≈ (φ(u) − φ(v)) × r_pre     (paper Eqs. 7–9)

Two-pass relaxation (paper §3.4, MNIST):
  1. Bottom-up: set each soma to its basal prediction.
  2. Nudge output somata toward 1-hot target rates {0.1, 0.8}.
  3. Reverse: SST (weak top-down teach) → apical PE after cancel → hidden soma mix.
  4. Local PE × pre updates on basal (and optional PI/IP) synapses.

Intel Lava's learning DSL is STDP/R-STDP, not PE×pre. This module is the
MNIST trainer. See ``sacramento.lava_microcircuit`` for the Lava Process that
demonstrates 3-compartment + SST-like cancel on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def logistic(u: np.ndarray) -> np.ndarray:
    """φ(u) = 1 / (1 + exp(−u)), clipped for overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(u, -60.0, 60.0)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def target_rates(labels: np.ndarray, n_classes: int, lo: float = 0.1, hi: float = 0.8) -> np.ndarray:
    """Paper-style 1-hot rates r ∈ {lo, hi}."""
    t = np.full((labels.shape[0], n_classes), lo, dtype=np.float64)
    t[np.arange(labels.shape[0]), labels.astype(np.int64)] = hi
    return t


@dataclass
class SacramentoConfig:
    layer_sizes: Tuple[int, ...] = (784, 256, 10)
    eta_pp: float = 0.05
    eta_ip: float = 0.0
    eta_pi: float = 0.0
    eta_bias: float = 0.05
    lambda_hidden: float = 0.3
    lambda_out: float = 0.5
    lambda_sst: float = 0.5
    basal_sparsity: float = 1.0
    apical_sparsity: float = 1.0
    seed: int = 0
    weight_scale: float = 3.0
    target_lo: float = 0.1
    target_hi: float = 0.8


class SacramentoRateNet:
    """Multi-layer pyramidal + SST-like cancel, local PE plasticity only."""

    def __init__(self, cfg: SacramentoConfig):
        if len(cfg.layer_sizes) < 2:
            raise ValueError("Need at least input and output sizes")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        sizes = list(cfg.layer_sizes)
        self.sizes = sizes
        self.n_hidden_layers = len(sizes) - 2  # layers that have apical + SST

        self.W_pp: List[np.ndarray] = []
        self.bias: List[np.ndarray] = []
        self.basal_mask: List[np.ndarray] = []
        self.W_fb: List[Optional[np.ndarray]] = []
        self.W_pi: List[Optional[np.ndarray]] = []
        self.W_ip: List[Optional[np.ndarray]] = []
        self.apical_mask: List[Optional[np.ndarray]] = []
        # Fixed random SST teaching map when hidden size ≠ next-layer size
        # (paper is 1:1 SST↔upper pyr). Here SST dim = next layer (matched),
        # so the projector is identity. Kept for API / residual documentation.
        self.teach_proj: List[Optional[np.ndarray]] = []

        for i in range(len(sizes) - 1):
            n_pre, n_post = sizes[i], sizes[i + 1]
            scale = cfg.weight_scale / np.sqrt(max(n_pre, 1))
            w = self.rng.normal(0.0, scale, size=(n_post, n_pre)).astype(np.float64)
            mask = self._sparsity_mask((n_post, n_pre), cfg.basal_sparsity)
            w *= mask
            self.W_pp.append(w)
            self.basal_mask.append(mask)
            self.bias.append(np.zeros(n_post, dtype=np.float64))

        # Hidden layers only: top-down FA weights, SST cancel (W_PI ≈ −W_fb)
        for h in range(self.n_hidden_layers):
            layer_idx = h + 1  # pyramidal layer index in `sizes`
            n_hid = sizes[layer_idx]
            n_next = sizes[layer_idx + 1]
            fb_scale = cfg.weight_scale / np.sqrt(max(n_next, 1))
            w_fb = self.rng.normal(0.0, fb_scale, size=(n_hid, n_next)).astype(np.float64)
            a_mask = self._sparsity_mask((n_hid, n_next), cfg.apical_sparsity)
            w_fb *= a_mask
            # Self-predicting cancel: W_PI = −W_fb so apical≈0 when r_SST ≈ r_next.
            # Frozen in the main MNIST setting (η_PI = 0). Residual PE is expected.
            w_pi = -w_fb.copy()
            # SST dendrites track next-layer basal (copy of W_pp into next layer).
            w_ip = self.W_pp[layer_idx].copy()
            self.W_fb.append(w_fb)
            self.W_pi.append(w_pi)
            self.W_ip.append(w_ip)
            self.apical_mask.append(a_mask)
            self.teach_proj.append(np.eye(n_next, dtype=np.float64))

    def _sparsity_mask(self, shape: Tuple[int, int], keep: float) -> np.ndarray:
        keep = float(keep)
        if keep >= 1.0:
            return np.ones(shape, dtype=np.float64)
        if keep <= 0.0:
            return np.zeros(shape, dtype=np.float64)
        return (self.rng.random(shape) < keep).astype(np.float64)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """Basal-only inference (no output nudge, no apical mix)."""
        r = np.asarray(x, dtype=np.float64)
        if r.ndim == 1:
            r = r[None, :]
        for w, b in zip(self.W_pp, self.bias):
            v_b = r @ w.T + b
            r = logistic(v_b)
        return r

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        rates = self.feedforward(x)
        pred = np.argmax(rates, axis=1)
        return float(np.mean(pred == np.asarray(y)))

    def _teach_batch(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Two-pass relaxation + local PE×pre deltas (not applied)."""
        cfg = self.cfg
        r0 = np.asarray(x, dtype=np.float64)
        if r0.ndim == 1:
            r0 = r0[None, :]
        bsz = r0.shape[0]
        y = np.asarray(y, dtype=np.int64)

        rates: List[np.ndarray] = [r0]
        v_basal: List[np.ndarray] = []
        u_soma: List[np.ndarray] = []

        # --- Pass 1: bottom-up basal predictions ---
        r = r0
        for w, b in zip(self.W_pp, self.bias):
            v_b = r @ w.T + b
            u = v_b.copy()
            r = logistic(u)
            v_basal.append(v_b)
            u_soma.append(u)
            rates.append(r)

        # Nudge output soma toward target voltages (logit of {0.1, 0.8} rates)
        n_out = self.sizes[-1]
        r_tgt = target_rates(y, n_out, cfg.target_lo, cfg.target_hi)
        u_tgt = logit(r_tgt)
        u_soma[-1] = (1.0 - cfg.lambda_out) * v_basal[-1] + cfg.lambda_out * u_tgt
        rates[-1] = logistic(u_soma[-1])

        # --- Pass 2: reverse SST / apical / hidden soma ---
        sst_u: List[Optional[np.ndarray]] = [None] * self.n_hidden_layers
        sst_r: List[Optional[np.ndarray]] = [None] * self.n_hidden_layers
        v_apical: List[Optional[np.ndarray]] = [None] * self.n_hidden_layers

        for h in range(self.n_hidden_layers - 1, -1, -1):
            # rates[h+1] is hidden pyramidal layer; rates[h+2] is the next layer
            r_pyr = rates[h + 1]
            u_next = u_soma[h + 1]
            r_next = rates[h + 2]
            v_i = r_pyr @ self.W_ip[h].T
            # Weak top-down teach on SST (paper: 1:1 toward upper pyramidal voltage)
            teach = u_next @ self.teach_proj[h].T
            u_i = (1.0 - cfg.lambda_sst) * v_i + cfg.lambda_sst * teach
            r_i = logistic(u_i)
            v_a = r_next @ self.W_fb[h].T + r_i @ self.W_pi[h].T
            u_hid = (1.0 - cfg.lambda_hidden) * v_basal[h] + cfg.lambda_hidden * v_a
            rates[h + 1] = logistic(u_hid)
            u_soma[h] = u_hid
            sst_u[h] = u_i
            sst_r[h] = r_i
            v_apical[h] = v_a

        # Local PE: φ(soma) − φ(basal prediction); pre is the bottom-up presynaptic rate
        # (input / pre-reverse hidden). Output PE uses the pre-update hidden rate.
        pes: List[np.ndarray] = []
        dW_pp: List[np.ndarray] = []
        d_bias: List[np.ndarray] = []

        r_pre_list = [rates[0]]
        # hidden rates before reverse were overwritten; reconstruct basal-only hidden
        r_hidden_bu = logistic(v_basal[0]) if self.n_hidden_layers else None
        if r_hidden_bu is not None:
            r_pre_list.append(r_hidden_bu)
        for i in range(1, len(self.W_pp) - 1):
            r_pre_list.append(logistic(v_basal[i]))

        for i, w in enumerate(self.W_pp):
            pe = logistic(u_soma[i]) - logistic(v_basal[i])
            pes.append(pe)
            r_pre = r_pre_list[i]
            dW_pp.append((pe.T @ r_pre) / bsz)
            d_bias.append(pe.mean(axis=0))

        dW_ip: List[Optional[np.ndarray]] = [None] * self.n_hidden_layers
        dW_pi: List[Optional[np.ndarray]] = [None] * self.n_hidden_layers
        for h in range(self.n_hidden_layers):
            r_pyr_bu = logistic(v_basal[h])
            v_i = r_pyr_bu @ self.W_ip[h].T
            # Eq. 8 attenuation ≈ (1 − λ_sst) on the SST dendrite
            hat_v_i = (1.0 - cfg.lambda_sst) * v_i
            pe_ip = logistic(sst_u[h]) - logistic(hat_v_i)
            dW_ip[h] = (pe_ip.T @ r_pyr_bu) / bsz
            # Eq. 9: silence apical (v_rest = 0)
            dW_pi[h] = ((0.0 - v_apical[h]).T @ sst_r[h]) / bsz

        mean_abs_pe = float(np.mean([np.mean(np.abs(p)) for p in pes]))
        return {
            "dW_pp": dW_pp,
            "d_bias": d_bias,
            "dW_ip": dW_ip,
            "dW_pi": dW_pi,
            "mean_abs_pe": mean_abs_pe,
            "pes": pes,
        }

    def _apply(self, stats: Dict[str, np.ndarray]) -> float:
        cfg = self.cfg
        abs_dw = 0.0
        n_w = 0
        for i, w in enumerate(self.W_pp):
            dw = cfg.eta_pp * stats["dW_pp"][i]
            w += dw
            w *= self.basal_mask[i]
            self.bias[i] += cfg.eta_bias * stats["d_bias"][i]
            abs_dw += float(np.sum(np.abs(dw)))
            n_w += dw.size
        if cfg.eta_ip != 0.0:
            for h in range(self.n_hidden_layers):
                self.W_ip[h] += cfg.eta_ip * stats["dW_ip"][h]
        if cfg.eta_pi != 0.0:
            for h in range(self.n_hidden_layers):
                self.W_pi[h] += cfg.eta_pi * stats["dW_pi"][h]
                if self.apical_mask[h] is not None:
                    # Keep PI support aligned with sparse apical / feedback.
                    fb_mask = self.apical_mask[h]
                    self.W_pi[h] *= fb_mask
                    self.W_fb[h] *= fb_mask
        return abs_dw / max(n_w, 1)

    def train_epoch(self, x: np.ndarray, y: np.ndarray, batch_size: int = 10) -> Dict[str, float]:
        n = x.shape[0]
        order = self.rng.permutation(n)
        pe_acc = 0.0
        dw_acc = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            stats = self._teach_batch(x[idx], y[idx])
            mean_dw = self._apply(stats)
            pe_acc += float(stats["mean_abs_pe"])
            dw_acc += mean_dw
            n_batches += 1
        train_acc = self.evaluate(x, y)
        return {
            "train_acc": train_acc,
            "mean_abs_pe": pe_acc / max(n_batches, 1),
            "mean_abs_dw_pp": dw_acc / max(n_batches, 1),
        }

    def mean_teach_pe(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32) -> float:
        n = x.shape[0]
        acc = 0.0
        nb = 0
        for start in range(0, n, batch_size):
            stats = self._teach_batch(x[start : start + batch_size], y[start : start + batch_size])
            acc += float(stats["mean_abs_pe"])
            nb += 1
        return acc / max(nb, 1)


def pe_learning_smoke(seed: int = 1, epochs: int = 40) -> Dict[str, float]:
    """Synthetic 8→6→3 proof that PE×pre moves weights and reduces teaching PE."""
    rng = np.random.default_rng(seed)
    n, din, hid, dout = 32, 8, 6, 3
    x = rng.random((n, din))
    y = rng.integers(0, dout, size=n)
    cfg = SacramentoConfig(
        layer_sizes=(din, hid, dout),
        eta_pp=0.2,
        eta_ip=0.0,
        eta_pi=0.0,
        eta_bias=0.2,
        lambda_hidden=0.4,
        lambda_out=0.5,
        lambda_sst=0.5,
        basal_sparsity=1.0,
        apical_sparsity=1.0,
        seed=seed,
        weight_scale=3.0,
    )
    net = SacramentoRateNet(cfg)
    w0 = [w.copy() for w in net.W_pp]
    pe_before = net.mean_teach_pe(x, y)
    last_acc = 0.0
    for _ in range(epochs):
        tr = net.train_epoch(x, y, batch_size=8)
        last_acc = tr["train_acc"]
    pe_after = net.mean_teach_pe(x, y)
    weight_delta_l1 = float(sum(np.sum(np.abs(a - b)) for a, b in zip(net.W_pp, w0)))
    return {
        "pe_before": float(pe_before),
        "pe_after": float(pe_after),
        "weight_delta_l1": weight_delta_l1,
        "train_acc_last": float(last_acc),
    }
