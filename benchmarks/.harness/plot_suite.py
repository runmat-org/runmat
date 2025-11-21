#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
from matplotlib import ticker

PRIMARY_BG = "#050505"
PANEL_BG = "#0e0e0e"
TEXT_COLOR = "#f2f2f2"
GRID_COLOR = "#373737"
NUMPY_COLOR = "#8f95a8"
RUNMAT_COLOR = "#b38aff"
PYTORCH_COLOR = "#5ad0ff"
EXPORT_EXT = ".svg"


def load_suite(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def style_axes(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, linestyle=":", linewidth=0.7, color=GRID_COLOR, alpha=0.7)


def apply_log_x(ax, log_x_min: Optional[float]) -> None:
    ax.set_xscale("log")
    if log_x_min:
        ax.set_xlim(left=log_x_min)
    major_locator = ticker.LogLocator(base=10.0, numticks=8)
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    minor_locator = ticker.LogLocator(base=10.0, subs=tuple(range(2, 10)))
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def plot_case(
    case: Dict[str, Any],
    out_dir: Path,
    log_x: bool = False,
    log_x_min: Optional[float] = None,
) -> None:
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
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.patch.set_facecolor(PRIMARY_BG)
    colors = {"runmat": RUNMAT_COLOR, "python-numpy": NUMPY_COLOR, "python-torch": PYTORCH_COLOR}
    markers = {"runmat": "o", "python-numpy": "^", "python-torch": "s"}
    for impl, pts in series.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs,
            ys,
            marker=markers.get(impl, "o"),
            label=impl,
            color=colors.get(impl),
            linewidth=2.2,
            markersize=6,
        )
    ax.set_xlabel(f"{x_param}")
    ax.set_ylabel("Median time (ms)")
    ax.set_title(f"{case.get('label', case.get('id'))} - scaling")
    if log_x:
        apply_log_x(ax, log_x_min)
    style_axes(ax)
    legend = ax.legend(frameon=False, labelcolor=TEXT_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{case.get('id')}_scaling{EXPORT_EXT}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Speedup plot vs numpy
    numpy_pts = {n: ms for (n, ms, _) in series.get("python-numpy", [])}
    if numpy_pts:
        fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
        fig2.patch.set_facecolor(PRIMARY_BG)
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
                ax2.plot(
                    xs,
                    ys,
                    marker=markers.get(impl, "o"),
                    label=impl,
                    color=colors.get(impl),
                    linewidth=2.2,
                    markersize=6,
                )
        ax2.axhline(1.0, color="#6c6c6c", linestyle="--", linewidth=1.2, label="numpy parity")
        ax2.set_xlabel(f"{x_param}")
        ax2.set_ylabel("Speedup vs numpy (x)")
        ax2.set_title(f"{case.get('label', case.get('id'))} - speedup")
        if log_x:
            apply_log_x(ax2, log_x_min)
        style_axes(ax2)
        legend2 = ax2.legend(frameon=False, labelcolor=TEXT_COLOR)
        for text in legend2.get_texts():
            text.set_color(TEXT_COLOR)
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{case.get('id')}_speedup{EXPORT_EXT}", dpi=180, facecolor=fig2.get_facecolor())
        plt.close(fig2)

    summary = case.get("summary") or {}
    telemetry_summary = summary.get("telemetry")
    if telemetry_summary:
        plot_telemetry(case, telemetry_summary, x_param, out_dir, log_x=log_x, log_x_min=log_x_min)


def plot_telemetry(
    case: Dict[str, Any],
    telemetry_summary: Dict[str, Any],
    x_param: str,
    out_dir: Path,
    log_x: bool = False,
    log_x_min: Optional[float] = None,
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

    height = 8 if has_kernel_table else 6.5
    fig, axes = plt.subplots(nrows, 1, figsize=(7.5, height))
    fig.patch.set_facecolor(PRIMARY_BG)
    if nrows == 1:
        axes = [axes]

    ax_up = axes[0]
    offsets = [p - 0.2 for p in params]
    ax_up.bar(offsets, upload, width=0.4, label="upload (MB)", color=NUMPY_COLOR)
    ax_up.bar([p + 0.2 for p in params], download, width=0.4, label="download (MB)", color=RUNMAT_COLOR)
    ax_up.set_xlabel(x_param)
    ax_up.set_ylabel("Data (MB)")
    ax_up.set_title(f"{case.get('label', case.get('id'))} - transfer telemetry")
    if log_x:
        apply_log_x(ax_up, log_x_min)
    style_axes(ax_up)
    legend_up = ax_up.legend(frameon=False, labelcolor=TEXT_COLOR)
    for text in legend_up.get_texts():
        text.set_color(TEXT_COLOR)

    ax_gpu = axes[1]
    ax_gpu.bar(params, fe_ms, width=0.6, label="fused elem", color=RUNMAT_COLOR)
    bottom = fe_ms
    fr_stack = [fr_ms[i] + bottom[i] for i in range(len(bottom))]
    ax_gpu.bar(params, fr_ms, width=0.6, bottom=bottom, label="fused reduction", color=PYTORCH_COLOR)
    mm_bottom = [bottom[i] + fr_ms[i] for i in range(len(bottom))]
    ax_gpu.bar(params, mm_ms, width=0.6, bottom=mm_bottom, label="matmul", color=NUMPY_COLOR)
    ax_gpu.set_xlabel(x_param)
    ax_gpu.set_ylabel("GPU wall time (ms)")
    ax_gpu.set_title("GPU kernel time by category")
    if log_x:
        apply_log_x(ax_gpu, log_x_min)
    style_axes(ax_gpu)
    legend_gpu = ax_gpu.legend(frameon=False, labelcolor=TEXT_COLOR)
    for text in legend_gpu.get_texts():
        text.set_color(TEXT_COLOR)

    if has_kernel_table:
        ax_table = axes[2]
        ax_table.axis("off")
        table = ax_table.table(
            cellText=kernel_rows,
            colLabels=["param", "lane", "spatial", "batch", "vals/thread"],
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax_table.set_title("ImageNormalize autotune selections", color=TEXT_COLOR)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{case.get('id')}_telemetry{EXPORT_EXT}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot suite results")
    ap.add_argument("--input", default=str(Path("../results/suite_results.json")))
    ap.add_argument("--output_dir", default=str(Path("../results")))
    ap.add_argument(
        "--log_x",
        action="store_true",
        help="Render plots with a logarithmic x-axis",
    )
    ap.add_argument(
        "--log_x_min",
        type=float,
        default=None,
        help="Optional minimum x value when log scale is enabled",
    )
    args = ap.parse_args()

    data = load_suite(Path(args.input))
    out_dir = Path(args.output_dir)

    for case in data.get("cases", []):
        case_plot = case.get("plot", {})
        case_log_x = case_plot.get("log_x")
        case_log_x_min = case_plot.get("log_x_min")
        # CLI flags override when explicitly provided; otherwise use per-case defaults.
        effective_log_x = args.log_x or bool(case_log_x)
        effective_log_x_min = args.log_x_min
        if effective_log_x_min is None:
            effective_log_x_min = case_log_x_min
        plot_case(case, out_dir, log_x=effective_log_x, log_x_min=effective_log_x_min)

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
