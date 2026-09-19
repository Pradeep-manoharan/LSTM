#!/usr/bin/env python3
"""CI / local smoke: PE×pre learning + cancel demo (Lava if importable)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sacramento.lava_microcircuit import run_lava_cancel_demo
from sacramento.rate_model import pe_learning_smoke


def main() -> int:
    smoke = pe_learning_smoke(seed=1)
    print("pe_learning_smoke:", json.dumps(smoke, indent=2))
    if not (smoke["weight_delta_l1"] > 0 and smoke["pe_after"] < smoke["pe_before"]):
        print("FAIL: expected PE×pre to move weights and reduce teaching PE")
        return 1

    cancel = run_lava_cancel_demo(n=4, steps=5, seed=0)
    matched = float(cancel["mean_abs_pe_matched"].reshape(-1)[0])
    mismatched = float(cancel["mean_abs_pe_mismatched"].reshape(-1)[0])
    print(
        f"cancel backend={cancel.get('backend')} "
        f"matched={matched:.6f} mismatched={mismatched:.6f}"
    )
    if mismatched <= matched:
        print("FAIL: expected mismatched |apical PE| > matched")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
