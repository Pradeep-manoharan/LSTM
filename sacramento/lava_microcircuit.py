"""3-compartment pyramidal + SST-like cancel as an Intel Lava Process.

This is a CPU ``Loihi1SimCfg`` structure demo (basal / soma / apical + cancel).
It is **not** the MNIST PE×pre trainer: Lava's built-in learning rules are
STDP / R-STDP. Full PE learning lives in ``sacramento.rate_model``.

If ``lava-nc`` cannot be imported (Python 3.13 / 3.12: no wheel; needs
CPython 3.10), ``run_lava_cancel_demo`` falls back to the same equations in
NumPy and labels the backend honestly.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def lava_nc_status() -> Dict[str, Any]:
    """Import probe used by docs / demos. Never invent a successful Lava run."""
    info: Dict[str, Any] = {
        "importable": False,
        "version": None,
        "error": None,
    }
    try:
        import importlib.metadata as md
        import lava  # noqa: F401

        info["importable"] = True
        try:
            info["version"] = md.version("lava-nc")
        except Exception as exc:  # pragma: no cover
            info["version"] = f"unknown ({exc})"
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _cancel_step(
    basal: np.ndarray,
    top_down: np.ndarray,
    sst: np.ndarray,
    w_pi: np.ndarray,
    lambda_apical: float,
) -> Dict[str, np.ndarray]:
    """One discrete cancel step (shared by Lava ProcessModel and NumPy)."""
    apical = top_down + w_pi @ sst
    soma = basal + lambda_apical * apical
    return {"apical": apical, "soma": soma}


def _numpy_cancel_demo(n: int, steps: int, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    basal = rng.normal(0.0, 0.4, size=n)
    top_down = rng.normal(0.3, 0.5, size=n)
    # Imperfect −I so matched PE is residual but smaller than mismatch (RESULTS-style).
    w_pi = -0.65 * np.eye(n)
    sst_matched = 1.0 / (1.0 + np.exp(-top_down))
    sst_mismatch = 1.0 / (1.0 + np.exp(-rng.normal(0.0, 0.8, size=n)))
    lambda_a = 0.35

    pe_m, pe_x = [], []
    last_m = last_x = None
    for _ in range(steps):
        last_m = _cancel_step(basal, top_down, sst_matched, w_pi, lambda_a)
        last_x = _cancel_step(basal, top_down, sst_mismatch, w_pi, lambda_a)
        pe_m.append(np.mean(np.abs(last_m["apical"])))
        pe_x.append(np.mean(np.abs(last_x["apical"])))

    return {
        "backend": "numpy-reference",
        "n": n,
        "steps": steps,
        "mean_abs_pe_matched": np.array([float(np.mean(pe_m))]),
        "mean_abs_pe_mismatched": np.array([float(np.mean(pe_x))]),
        "apical_matched": last_m["apical"],
        "apical_mismatched": last_x["apical"],
        "note": (
            "lava-nc not importable; NumPy used the same apical = top_down + W_PI @ SST "
            "cancel equation. Install lava-nc on CPython 3.10 to run the Process."
        ),
    }


def _run_lava_process_pair(n: int, steps: int, seed: int = 0) -> Dict[str, Any]:
    from lava.magma.core.decorator import implements, requires, tag
    from lava.magma.core.model.py.model import PyLoihiProcessModel
    from lava.magma.core.model.py.type import LavaPyType
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.core.process.variable import Var
    from lava.magma.core.resources import CPU
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi1SimCfg
    from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

    class PyramidalSSTMicrocircuit(AbstractProcess):
        """Basal / soma / apical pyramidal with SST-like apical cancel."""

        def __init__(
            self,
            basal: np.ndarray,
            top_down: np.ndarray,
            sst: np.ndarray,
            w_pi: np.ndarray,
            lambda_apical: float = 0.35,
        ):
            super().__init__()
            n_u = int(basal.shape[0])
            self.basal = Var(shape=(n_u,), init=np.asarray(basal, dtype=float))
            self.soma = Var(shape=(n_u,), init=np.zeros(n_u, dtype=float))
            self.apical = Var(shape=(n_u,), init=np.zeros(n_u, dtype=float))
            self.top_down = Var(shape=(n_u,), init=np.asarray(top_down, dtype=float))
            self.sst = Var(shape=(n_u,), init=np.asarray(sst, dtype=float))
            self.w_pi = Var(shape=(n_u, n_u), init=np.asarray(w_pi, dtype=float))
            self.lambda_apical = Var(shape=(1,), init=float(lambda_apical))

    @implements(proc=PyramidalSSTMicrocircuit, protocol=LoihiProtocol)
    @requires(CPU)
    @tag("floating_pt")
    class PyPyramidalSSTMicrocircuit(PyLoihiProcessModel):
        basal: np.ndarray = LavaPyType(np.ndarray, float)
        soma: np.ndarray = LavaPyType(np.ndarray, float)
        apical: np.ndarray = LavaPyType(np.ndarray, float)
        top_down: np.ndarray = LavaPyType(np.ndarray, float)
        sst: np.ndarray = LavaPyType(np.ndarray, float)
        w_pi: np.ndarray = LavaPyType(np.ndarray, float)
        lambda_apical: float = LavaPyType(float, float)

        def run_spk(self):
            apical = self.top_down + self.w_pi @ self.sst
            self.apical[:] = apical
            self.soma[:] = self.basal + float(self.lambda_apical) * apical

    rng = np.random.default_rng(seed)
    basal = rng.normal(0.0, 0.4, size=n)
    top_down = rng.normal(0.3, 0.5, size=n)
    w_pi = -0.65 * np.eye(n)
    sst_matched = 1.0 / (1.0 + np.exp(-top_down))
    sst_mismatch = 1.0 / (1.0 + np.exp(-rng.normal(0.0, 0.8, size=n)))
    lambda_a = 0.35
    run_cfg = Loihi1SimCfg(select_tag="floating_pt")

    def _run(sst: np.ndarray) -> np.ndarray:
        proc = PyramidalSSTMicrocircuit(
            basal=basal, top_down=top_down, sst=sst, w_pi=w_pi, lambda_apical=lambda_a
        )
        proc.run(condition=RunSteps(num_steps=steps), run_cfg=run_cfg)
        apical = np.asarray(proc.apical.get(), dtype=np.float64)
        proc.stop()
        return apical

    apical_m = _run(sst_matched)
    apical_x = _run(sst_mismatch)
    return {
        "backend": "lava-nc",
        "n": n,
        "steps": steps,
        "mean_abs_pe_matched": np.array([float(np.mean(np.abs(apical_m)))]),
        "mean_abs_pe_mismatched": np.array([float(np.mean(np.abs(apical_x)))]),
        "apical_matched": apical_m,
        "apical_mismatched": apical_x,
        "note": "CPU Loihi1SimCfg Process: apical = top_down + W_PI @ SST",
    }


def run_lava_cancel_demo(n: int = 4, steps: int = 5, seed: int = 0) -> Dict[str, Any]:
    """Compare matched vs mismatched SST teach; mismatched should yield larger |PE|."""
    status = lava_nc_status()
    if status["importable"]:
        out = _run_lava_process_pair(n=n, steps=steps, seed=seed)
        out["lava_nc_version"] = status["version"]
        return out
    out = _numpy_cancel_demo(n=n, steps=steps, seed=seed)
    out["lava_import_error"] = status["error"]
    return out
