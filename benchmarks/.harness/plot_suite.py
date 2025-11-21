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

    summary = case.get("summary") or {}
    telemetry_summary = summary.get("telemetry")
    if telemetry_summary:
        plot_telemetry(case, telemetry_summary, x_param, out_dir)


def plot_telemetry(
    case: Dict[str, Any],
    telemetry_summary: Dict[str, Any],
    x_param: str,
    out_dir: Path,
) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    per_point = telemetry_summary.get("per_point") or []
    if not per_point:
        return

    records: List[Dict[str, Any]] = []
    for pt in per_point:
        param_entry = pt.get("param") or {}
        val = param_entry.get(x_param)
        if val is None:
            continue
        try:
            pval = int(val)
        except Exception:
            continue
        records.append(
            {
                "param": pval,
                "upload": float(pt.get("upload_bytes") or 0) / 1_000_000.0,
                "download": float(pt.get("download_bytes") or 0) / 1_000_000.0,
                "fe_ms": float((pt.get("fused_elementwise") or {}).get("wall_ms") or 0.0),
                "fr_ms": float((pt.get("fused_reduction") or {}).get("wall_ms") or 0.0),
                "mm_ms": float((pt.get("matmul") or {}).get("wall_ms") or 0.0),
                "kernel_launches": pt.get("kernel_launches") or [],
            }
        )

    if not records:
        return

    records.sort(key=lambda r: r["param"])
    params = [r["param"] for r in records]
    upload = [r["upload"] for r in records]
    download = [r["download"] for r in records]
    fe_ms = [r["fe_ms"] for r in records]
    fr_ms = [r["fr_ms"] for r in records]
    mm_ms = [r["mm_ms"] for r in records]

    kernel_rows: List[List[Any]] = []
    for r in records:
        launches = r["kernel_launches"]
        if not launches:
            continue
        image_launch = next(
            (launch for launch in launches if (launch.get("kernel") or "").lower() == "image_normalize"),
            None,
        )
        if not image_launch:
            continue
        tuning_entries = image_launch.get("tuning") or []
        tuning_map = {str(item.get("key")): item.get("value") for item in tuning_entries if isinstance(item, dict)}
        kernel_rows.append(
            [
                r["param"],
                tuning_map.get("lane_count"),
                tuning_map.get("spatial_tile"),
                tuning_map.get("batch_tile"),
                tuning_map.get("values_per_thread"),
            ]
        )

    has_kernel_table = bool(kernel_rows)
    nrows = 3 if has_kernel_table else 2

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    height = 8 if has_kernel_table else 6
    fig, axes = plt.subplots(nrows, 1, figsize=(7, height))
    if nrows == 1:
        axes = [axes]

    ax_up = axes[0]
    offsets = [p - 0.2 for p in params]
    ax_up.bar(offsets, upload, width=0.4, label="upload (MB)", color="#4e79a7")
    ax_up.bar([p + 0.2 for p in params], download, width=0.4, label="download (MB)", color="#f28e2b")
    ax_up.set_xlabel(x_param)
    ax_up.set_ylabel("Data (MB)")
    ax_up.set_title(f"{case.get('label', case.get('id'))} - transfer telemetry")
    ax_up.grid(True, linestyle=":", alpha=0.4)
    ax_up.legend()

    ax_gpu = axes[1]
    ax_gpu.bar(params, fe_ms, width=0.6, label="fused elem", color="#59a14f")
    bottom = fe_ms
    fr_stack = [fr_ms[i] + bottom[i] for i in range(len(bottom))]
    ax_gpu.bar(params, fr_ms, width=0.6, bottom=bottom, label="fused reduction", color="#edc948")
    mm_bottom = [bottom[i] + fr_ms[i] for i in range(len(bottom))]
    ax_gpu.bar(params, mm_ms, width=0.6, bottom=mm_bottom, label="matmul", color="#b07aa1")
    ax_gpu.set_xlabel(x_param)
    ax_gpu.set_ylabel("GPU wall time (ms)")
    ax_gpu.set_title("GPU kernel time by category")
    ax_gpu.grid(True, linestyle=":", alpha=0.4)
    ax_gpu.legend()

    if has_kernel_table:
        ax_table = axes[2]
        ax_table.axis("off")
        table = ax_table.table(
            cellText=kernel_rows,
            colLabels=["param", "lane", "spatial", "batch", "vals/thread"],
            loc="center",
        )
        table.scale(1, 1.5)
        ax_table.set_title("ImageNormalize autotune selections")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{case.get('id')}_telemetry.png", dpi=150)


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
