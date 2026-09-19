"""Module-level Lava Process / ProcessModel (compiler discovery requires this)."""

from __future__ import annotations

import numpy as np
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
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
