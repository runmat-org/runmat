#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

def main() -> None:
    ap = argparse.ArgumentParser(description="Plot benchmark results as normalized bar chart")
    ap.add_argument("--input", required=True, help="results JSON from run_bench.py")
    ap.add_argument("--output", required=True, help="output PNG path")
    ap.add_argument("--baseline", default="python-numpy", help="implementation name used as baseline (normalized to 1×)")
    args = ap.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    name_to_ms = {r["impl"]: float(r["median_ms"]) for r in results if r.get("median_ms") is not None}

    if args.baseline not in name_to_ms:
        raise SystemExit(f"Baseline '{args.baseline}' not found in results: {list(name_to_ms.keys())}")

    baseline_ms = name_to_ms[args.baseline]

    preferred_order = [
        "python-numpy",
        "octave",
        "julia",
        "python-torch",
        "runmat",
    ]
    impls = [n for n in preferred_order if n in name_to_ms]
    values = [baseline_ms / name_to_ms[n] for n in impls]  # speedup × vs loops

    # Use non-interactive backend and create plot
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(impls, values, color=["#4e79a7", "#888", "#e15759", "#59a14f", "#f28e2b"][: len(values)])
    ax.set_ylabel("× vs NumPy (higher is faster)")
    ax.set_title(f"{data.get('case', 'benchmark')} — Relative speed")
    ax.set_ylim(0, max(values) * 1.25 if values else 1)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1f}×", ha="center", va="bottom", fontsize=9)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()