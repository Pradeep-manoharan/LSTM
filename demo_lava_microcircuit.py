#!/usr/bin/env python3
"""Run Lava CPU simulation of 3-compartment + SST cancel microcircuit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sacramento.lava_microcircuit import lava_nc_status, run_lava_cancel_demo


def main():
    print("=== Lava microcircuit SST-cancel demo ===")
    status = lava_nc_status()
    print(f"lava-nc importable={status['importable']} version={status['version']}")
    if status["error"]:
        print("lava-nc import error:", status["error"])

    out = run_lava_cancel_demo(n=4, steps=5)
    summary = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in out.items()
    }
    print(json.dumps(summary, indent=2))
    matched = float(np.asarray(out["mean_abs_pe_matched"]).reshape(-1)[0])
    mismatched = float(np.asarray(out["mean_abs_pe_mismatched"]).reshape(-1)[0])
    print(f"backend: {out.get('backend')}")
    print(f"mean_|PE| matched teach:     {matched:.6f}")
    print(f"mean_|PE| mismatched teach:  {mismatched:.6f}")
    if mismatched > matched:
        print("OK: mismatched top-down/SST produces larger apical PE (cancel works directionally).")
    else:
        print("NOTE: expected mismatched PE > matched PE; inspect weights/teach encoding.")

    art = ROOT / "artifacts" / "lava_cancel_demo.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    with open(art, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {art}")


if __name__ == "__main__":
    main()
