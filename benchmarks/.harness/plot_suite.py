#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib


def load_suite(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def plot_case(case: Dict[str, Any], out_dir: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    results = case.get("results", [])
    # Determine x param
    sweep = case.get("sweep", {})
    x_param = sweep.get("param") or "n"
    # Group by impl -> [(x, ms, stdout)]
    series: Dict[str, List[tuple]] = {}
    for r in results:
        impl = r.get("impl")
        n = r.get(x_param)
        ms = r.get("median_ms")
        if impl is None or n is None or ms is None:
            continue
        series.setdefault(impl, []).append((int(n), float(ms), r.get("stdout_tail", "")))
    for v in series.values():
        v.sort(key=lambda t: t[0])

    # Scaling plot (ms vs n)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"runmat": "#f28e2b", "python-numpy": "#4e79a7", "python-torch": "#59a14f"}
    markers = {"runmat": "o", "python-numpy": "^", "python-torch": "s"}
    for impl, pts in series.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=markers.get(impl, "o"), label=impl, color=colors.get(impl))
    ax.set_xlabel(f"{x_param}")
    ax.set_ylabel("Median time (ms)")
    ax.set_title(f"{case.get('label', case.get('id'))} - scaling")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{case.get('id')}_scaling.png", dpi=150)

    # Speedup plot vs numpy
    numpy_pts = {n: ms for (n, ms, _) in series.get("python-numpy", [])}
    if numpy_pts:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        for impl, pts in series.items():
            if impl == "python-numpy":
                continue
            xs = []
            ys = []
            for n, ms, _ in pts:
                base = numpy_pts.get(n)
                if base:
                    xs.append(n)
                    ys.append(base / ms)
            if xs:
                ax2.plot(xs, ys, marker=markers.get(impl, "o"), label=impl, color=colors.get(impl))
        ax2.axhline(1.0, color="#999", linestyle=":", label="numpy parity")
        ax2.set_xlabel("n (rows)")
        ax2.set_ylabel("Speedup vs numpy (x)")
        ax2.set_title(f"{case.get('label', case.get('id'))} - speedup")
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{case.get('id')}_speedup.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot suite results")
    ap.add_argument("--input", default=str(Path("../results/suite_results.json")))
    ap.add_argument("--output_dir", default=str(Path("../results")))
    args = ap.parse_args()

    data = load_suite(Path(args.input))
    out_dir = Path(args.output_dir)

    for case in data.get("cases", []):
        plot_case(case, out_dir)

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
