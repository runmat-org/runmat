#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .utils import default_env, ensure_output_path, median, parse_mse, run_cmd, which
except ImportError:
    # Allow running as a loose script: python3 harness/run_bench.py
    sys.path.append(str(Path(__file__).parent))
    from utils import default_env, ensure_output_path, median, parse_mse, run_cmd, which


ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = ROOT / "benchmarks" / "benchmarks"


def detect_runmat_command() -> List[str]:
    if which("runmat"):
        return ["runmat"]
    # Fallback to cargo run
    return ["cargo", "run", "-q", "-p", "runmat", "--release", "--"]


def build_impl_commands(case: str) -> List[Dict]:
    """Return a list of implementations with command invocations and metadata."""
    casedir = CASES_DIR / case
    impls: List[Dict] = []

    # RunMat
    runmat_m = casedir / "runmat.m"
    if runmat_m.exists():
        impls.append(
            {
                "name": "runmat",
                "lang": "matlab-syntax",
                "cmd": detect_runmat_command() + [str(runmat_m)],
            }
        )

    # GNU Octave (reuse the same .m file)
    if runmat_m.exists() and which("octave"):
        impls.append(
            {
                "name": "octave",
                "lang": "octave",
                "cmd": ["octave", "-qf", str(runmat_m)],
            }
        )

    # Python NumPy
    numpy_py = casedir / "python_numpy.py"
    if numpy_py.exists() and which("python3"):
        impls.append(
            {
                "name": "python-numpy",
                "lang": "python",
                "cmd": ["python3", str(numpy_py)],
            }
        )

    # Python PyTorch
    torch_py = casedir / "python_torch.py"
    if torch_py.exists() and which("python3"):
        impls.append(
            {
                "name": "python-torch",
                "lang": "python",
                "cmd": ["python3", str(torch_py)],
            }
        )

    # Julia
    julia_jl = casedir / "julia.jl"
    if julia_jl.exists() and which("julia"):
        impls.append(
            {
                "name": "julia",
                "lang": "julia",
                "cmd": ["julia", "--color=no", str(julia_jl)],
            }
        )

    return impls


def run_impl(impl: Dict, iterations: int, timeout: int) -> Dict:
    env = default_env()
    times_ms: List[float] = []
    mses: List[float] = []
    last_stdout = ""
    last_stderr = ""
    for _ in range(iterations):
        rc, elapsed_ms, out, err = run_cmd(impl["cmd"], env=env, timeout=timeout)
        last_stdout, last_stderr = out, err
        times_ms.append(elapsed_ms)
        mse = parse_mse(out)
        if mse is not None:
            mses.append(mse)
    result = {
        "impl": impl["name"],
        "lang": impl["lang"],
        "iterations": iterations,
        "median_ms": median(times_ms),
        "all_ms": times_ms,
        "mse": mses[-1] if mses else None,
        "stdout_tail": last_stdout[-500:],
        "stderr_tail": last_stderr[-500:],
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run cross-language benchmarks")
    ap.add_argument("--case", required=True, help="case directory name (e.g., 4k-image-processing)")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=0, help="per-run timeout (seconds), 0 = unlimited")
    ap.add_argument("--output", type=Path, default=Path("../results/results.json"))
    args = ap.parse_args()

    impls = build_impl_commands(args.case)
    if not impls:
        raise SystemExit(f"No implementations found for case: {args.case}")

    ensure_output_path(args.output)
    suite_result = {
        "case": args.case,
        "results": [],
    }
    for impl in impls:
        suite_result["results"].append(run_impl(impl, args.iterations, args.timeout))

    with open(args.output, "w") as f:
        json.dump(suite_result, f, indent=2)
    print(json.dumps(suite_result, indent=2))


if __name__ == "__main__":
    main()