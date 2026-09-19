"""Unit checks for local PE×pre (no MNIST download, no lava-nc)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sacramento.rate_model import SacramentoConfig, SacramentoRateNet, pe_learning_smoke


def test_pe_smoke_moves_weights_and_drops_pe():
    out = pe_learning_smoke(seed=1)
    assert out["weight_delta_l1"] > 0
    assert out["pe_after"] < out["pe_before"]


def test_sparsity_masks():
    cfg = SacramentoConfig(
        layer_sizes=(20, 12, 5),
        basal_sparsity=0.7,
        apical_sparsity=0.7,
        seed=0,
        weight_scale=3.0,
    )
    net = SacramentoRateNet(cfg)
    keep_b = float(np.mean(net.basal_mask[0] > 0))
    keep_a = float(np.mean(net.apical_mask[0] > 0))
    assert 0.4 < keep_b < 0.95
    assert 0.4 < keep_a < 0.95
    assert np.all(net.W_pp[0][net.basal_mask[0] == 0] == 0)


def test_evaluate_shape():
    cfg = SacramentoConfig(layer_sizes=(8, 6, 3), seed=2, weight_scale=3.0)
    net = SacramentoRateNet(cfg)
    x = np.random.default_rng(0).random((16, 8))
    y = np.zeros(16, dtype=np.int64)
    acc = net.evaluate(x, y)
    assert 0.0 <= acc <= 1.0


if __name__ == "__main__":
    test_pe_smoke_moves_weights_and_drops_pe()
    test_sparsity_masks()
    test_evaluate_shape()
    print("tests ok")
