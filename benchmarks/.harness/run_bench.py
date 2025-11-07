#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .utils import default_env, ensure_output_path, median, parse_mse, run_cmd, which
except ImportError:
    # Allow running as a loose script: python3 harness/run_bench.py
    sys.path.append(str(Path(__file__).parent))
    from utils import default_env, ensure_output_path, median, parse_mse, run_cmd, which


# Resolve repo root and cases dir robustly regardless of how this script is invoked.
# This file lives at: <repo>/benchmarks/.harness/run_bench.py
# So the repo root is two parents up.
ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = ROOT / "benchmarks"


def detect_runmat_command() -> List[str]:
    # Prefer prebuilt release binary with WGPU features
    bin_path = ROOT / "target" / "release" / "runmat"
    if bin_path.exists():
        return [str(bin_path)]
    # Next prefer a globally installed runmat
    if which("runmat"):
        return ["runmat"]
    # No implicit build here to avoid counting build time in runs
    raise SystemExit(
        "runmat binary not found. Please prebuild with:\n"
        "  cargo build -p runmat --release -F runmat-accelerate/wgpu -F runmat-runtime/wgpu\n"
        "or install runmat in PATH."
    )


def build_impl_commands(case: str, variant: str = "default") -> List[Dict]:
    """Return a list of implementations with command invocations and metadata.

    If variant != "default", prefer variant-specific files when present, e.g.:
      - runmat_{variant}.m
      - python_numpy_{variant}.py
      - python_torch_{variant}.py
      - julia_{variant}.jl
    Fallback to default filenames when a variant file does not exist.
    """
    casedir = CASES_DIR / case
    impls: List[Dict] = []

    # RunMat
    runmat_m_variant = casedir / f"runmat_{variant}.m"
    runmat_m_default = casedir / "runmat.m"
    runmat_m = runmat_m_variant if runmat_m_variant.exists() else runmat_m_default
    if runmat_m.exists():
        impls.append(
            {
                "name": "runmat",
                "lang": "matlab-syntax",
                "cmd": detect_runmat_command() + [str(runmat_m)],
                "script": str(runmat_m),
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

    # Helper: prefer repo-local venv if present for Python impls
    def detect_python_interpreter() -> List[str]:
        if os.name == "nt":
            p = ROOT / ".bench_venv" / "Scripts" / "python.exe"
        else:
            p = ROOT / ".bench_venv" / "bin" / "python"
        if p.exists():
            return [str(p)]
        return ["python3"] if which("python3") else ["python"]

    # Python NumPy
    numpy_py_variant = casedir / f"python_numpy_{variant}.py"
    numpy_py_default = casedir / "python_numpy.py"
    numpy_py = numpy_py_variant if numpy_py_variant.exists() else numpy_py_default
    if numpy_py.exists() and (which("python3") or (ROOT / ".bench_venv").exists()):
        impls.append(
            {
                "name": "python-numpy",
                "lang": "python",
                "cmd": detect_python_interpreter() + [str(numpy_py)],
                "script": str(numpy_py),
            }
        )

    # Python PyTorch
    torch_py_variant = casedir / f"python_torch_{variant}.py"
    torch_py_default = casedir / "python_torch.py"
    torch_py = torch_py_variant if torch_py_variant.exists() else torch_py_default
    if torch_py.exists() and (which("python3") or (ROOT / ".bench_venv").exists()):
        impls.append(
            {
                "name": "python-torch",
                "lang": "python",
                "cmd": detect_python_interpreter() + [str(torch_py)],
                "script": str(torch_py),
            }
        )

    # Julia
    julia_jl_variant = casedir / f"julia_{variant}.jl"
    julia_jl_default = casedir / "julia.jl"
    julia_jl = julia_jl_variant if julia_jl_variant.exists() else julia_jl_default
    if julia_jl.exists() and which("julia"):
        impls.append(
            {
                "name": "julia",
                "lang": "julia",
                "cmd": ["julia", "--color=no", str(julia_jl)],
            }
        )

    return impls


def _collect_runmat_telemetry(cmd: List[str], reset: bool = False) -> Optional[Dict]:
    if not cmd:
        return None
    accel_cmd = [cmd[0], "accel-info", "--json"]
    if reset:
        accel_cmd.append("--reset")
    rc, _, out, err = run_cmd(accel_cmd, env=default_env())
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except Exception:
        try:
            return json.loads(err)
        except Exception:
            return None


def _load_runmat_telemetry_file(path: Optional[Path]) -> Optional[Dict]:
    if path is None:
        return None
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def run_impl(impl: Dict, iterations: int, timeout: int, env_overrides: Optional[Dict[str, str]] = None) -> Dict:
    env = default_env()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})
    # For RunMat, optionally inject assignment prelude to avoid top-level if/parse issues
    if impl.get("name") == "runmat" and "script" in impl:
        assign = env.get("HARNESS_ASSIGN", "")
        if assign.strip():
            try:
                src = Path(impl["script"])  # type: ignore
                code = src.read_text()
                tmp_dir = ROOT / "target" / "bench_tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / (src.stem + "_assigned.m")
                tmp_path.write_text(assign + code)
                cmd_script = str(tmp_path)
                impl_cmd = list(impl["cmd"])  # type: ignore
                impl_cmd[-1] = cmd_script
                impl = {**impl, "cmd": impl_cmd}
            except Exception:
                pass
    times_ms: List[float] = []
    mses: List[float] = []
    last_stdout = ""
    last_stderr = ""
    telemetry_payload: Optional[Dict] = None
    telemetry_path: Optional[Path] = None
    if impl.get("name") == "runmat":
        tmp_dir = ROOT / "target" / "bench_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = tmp_dir / "provider_telemetry.json"
        try:
            telemetry_path.unlink()
        except FileNotFoundError:
            pass
        env["RUNMAT_TELEMETRY_OUT"] = str(telemetry_path)
        env["RUNMAT_TELEMETRY_RESET"] = "1"
        _collect_runmat_telemetry(impl["cmd"], reset=True)
    for _ in range(iterations):
        cmd = list(impl["cmd"])
        rc, elapsed_ms, out, err = run_cmd(cmd, env=env, timeout=timeout)
        last_stdout, last_stderr = out, err
        times_ms.append(elapsed_ms)
        mse = parse_mse(out)
        if mse is not None:
            mses.append(mse)
    # Try to capture reported device (if any)
    dev = None
    try:
        m = re.search(r"device=([A-Za-z0-9_:\\-]+)", last_stdout)
        if m:
            dev = m.group(1)
    except Exception:
        dev = None
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
    if dev:
        result["device"] = dev
    if impl.get("name") == "runmat":
        telemetry_payload = _load_runmat_telemetry_file(telemetry_path)
        if telemetry_payload is None:
            telemetry_payload = _collect_runmat_telemetry(impl["cmd"], reset=False)
        if telemetry_payload is not None:
            result["provider_telemetry"] = telemetry_payload
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run cross-language benchmarks")
    ap.add_argument("--case", required=True, help="case directory name (e.g., 4k-image-processing)")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=0, help="per-run timeout (seconds), 0 = unlimited")
    ap.add_argument("--output", type=Path, default=Path("../results/results.json"))
    ap.add_argument("--variant", type=str, default="default", help="variant suffix to prefer (e.g., 'small')")
    ap.add_argument("--sweep-n", type=str, default="", help="comma-separated list of n (rows) sizes to sweep, e.g. '10000,20000,40000,80000'")
    ap.add_argument("--pca-d", type=int, default=0, help="override d (cols) for PCA; 0 means use script default")
    ap.add_argument("--pca-k", type=int, default=0, help="override k (components) for PCA; 0 means use script default")
    ap.add_argument("--pca-iters", type=int, default=0, help="override iteration count; 0 means use script default")
    ap.add_argument(
        "--include-impl",
        type=str,
        default="",
        help="comma-separated list of impl names to include (e.g., runmat,python-numpy). If set, only these will run.",
    )
    ap.add_argument(
        "--exclude-impl",
        type=str,
        default="",
        help="comma-separated list of impl names to exclude (e.g., octave)",
    )
    args = ap.parse_args()

    impls = build_impl_commands(args.case, args.variant)
    if not impls:
        raise SystemExit(f"No implementations found for case: {args.case}")

    include: List[str] = [s.strip() for s in args.include_impl.split(",") if s.strip()]
    exclude: List[str] = [s.strip() for s in args.exclude_impl.split(",") if s.strip()]

    if include:
        impls = [impl for impl in impls if impl["name"] in include]
        if not impls:
            raise SystemExit(f"No implementations remain after include filter: {include}")
    if exclude:
        impls = [impl for impl in impls if impl["name"] not in exclude]
        if not impls:
            raise SystemExit(f"No implementations remain after exclude filter: {exclude}")

    ensure_output_path(args.output)

    sweep_values: List[int] = []
    if args.sweep_n.strip():
        try:
            sweep_values = [int(s.strip()) for s in args.sweep_n.split(",") if s.strip()]
        except Exception:
            raise SystemExit(f"Invalid --sweep-n: {args.sweep_n}")

    if sweep_values:
        suite_result = {
            "case": args.case,
            "sweep": {
                "param": "n",
                "values": sweep_values,
                "d": args.pca_d if args.pca_d else None,
                "k": args.pca_k if args.pca_k else None,
                "iters": args.pca_iters if args.pca_iters else None,
            },
            "results": [],
        }
        for nval in sweep_values:
            overrides: Dict[str, str] = {"PCA_N": str(nval)}
            if args.pca_d:
                overrides["PCA_D"] = str(args.pca_d)
            if args.pca_k:
                overrides["PCA_K"] = str(args.pca_k)
            if args.pca_iters:
                overrides["PCA_ITERS"] = str(args.pca_iters)
            for impl in impls:
                result = run_impl(impl, args.iterations, args.timeout, env_overrides=overrides)
                result["n"] = nval
                suite_result["results"].append(result)
    else:
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