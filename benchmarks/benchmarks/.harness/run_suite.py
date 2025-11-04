#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Local imports
try:
    from .utils import ensure_output_path, which
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from utils import ensure_output_path, which  # type: ignore

ROOT = Path(__file__).resolve().parents[3]
def _prebuild_runmat() -> None:
    bin_path = ROOT / "target" / "release" / "runmat"
    if bin_path.exists():
        return
    # Allow opt-out
    if os.environ.get("SKIP_PREBUILD_RUNMAT"):
        return
    cmd = [
        "cargo", "build", "-p", "runmat", "--release",
        "-F", "runmat-accelerate/wgpu", "-F", "runmat-runtime/wgpu",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if proc.returncode != 0 or not bin_path.exists():
        raise SystemExit("Failed to prebuild runmat release binary with WGPU features")

CASES_DIR = ROOT / "benchmarks"
HARNESS_DIR = CASES_DIR / ".harness"


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    if path.suffix in (".json",):
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit(
            f"PyYAML is required to parse {path.name}. Install with `python3 -m pip install pyyaml` or provide JSON.\nError: {e}"
        )
    return yaml.safe_load(text)  # type: ignore


essential_impls = ("runmat", "python-numpy", "python-torch")


def _extract_metric(stdout_tail: str, regex: str) -> Optional[float]:
    try:
        m = re.search(regex, stdout_tail)
        if not m:
            return None
        return float(m.group(1))
    except Exception:
        return None


def _run_bench(case: str, iterations: int, timeout: int, out_path: Path, variant: str, overrides: Dict[str, Any], include: List[str], extra_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(HARNESS_DIR / "run_bench.py"),
        "--case",
        case,
        "--iterations",
        str(iterations),
        "--timeout",
        str(timeout),
        "--variant",
        variant,
        "--output",
        str(out_path),
    ]
    if include:
        cmd += ["--include-impl", ",".join(include)]
    # PCA-specific parameterization via environment variables understood by scripts
    sweep_n = overrides.get("n")
    if isinstance(sweep_n, list) and sweep_n:
        cmd += ["--sweep-n", ",".join(str(x) for x in sweep_n)]
    # Optional PCA knobs
    for k in ("d", "k", "iters"):
        v = overrides.get(k)
        if isinstance(v, int) and v > 0:
            cmd += [f"--pca-{k}", str(v)]
    env = os.environ.copy()
    # Respect device prefs via env if provided in suite (no-ops if unused)
    device_env = overrides.get("env", {})
    for ek, ev in device_env.items():
        env[str(ek)] = str(ev)
    if extra_env:
        for ek, ev in extra_env.items():
            env[str(ek)] = str(ev)
    # Ensure headless
    env.setdefault("NO_GUI", "1")
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"run_bench failed for case={case}")
    try:
        return json.loads(proc.stdout)
    except Exception:
        return json.loads(out_path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full benchmark suite with parity checks")
    ap.add_argument("--suite", default=str(HARNESS_DIR / "suite.yaml"))
    ap.add_argument("--output", default=str(CASES_DIR / "../results/suite_results.json"))
    args = ap.parse_args()

    # Ensure runmat is prebuilt once so build time is not included in any run
    _prebuild_runmat()
    suite_path = Path(args.suite).resolve()
    cfg = _load_yaml_or_json(suite_path)

    global_cfg = cfg.get("global", {})
    iterations = int(global_cfg.get("iterations", 1))
    timeout_s = int(global_cfg.get("timeout_s", 0))

    results: Dict[str, Any] = {
        "suite": {
            "version": cfg.get("version", 1),
            "started_at": datetime.utcnow().isoformat() + "Z",
        },
        "cases": [],
    }

    for case in cfg.get("cases", []):
        case_id = case.get("id")
        if not case_id:
            continue
        label = case.get("label", case_id)
        case_dir = case.get("case", case_id)  # directory name under benchmarks/benchmarks
        parity = case.get("parity", {})
        metric_regex = parity.get("metric_regex")
        rtol = float(parity.get("rtol", 1e-3))
        atol = float(parity.get("atol", 1e-5))
        harness = case.get("harness", {})
        warmup = int(harness.get("warmup", 0))
        per_case_timeout = int(harness.get("timeout_s", timeout_s))

        # Build overrides for run_bench
        params = case.get("params", {})
        overrides: Dict[str, Any] = {}
        if "n" in params:
            overrides["n"] = list(params["n"])  # sweep list
        for k in ("d", "k", "iters"):
            if k in params:
                # single or list; we do single for now (first value)
                v = params[k]
                if isinstance(v, list):
                    overrides[k] = int(v[0])
                elif isinstance(v, int):
                    overrides[k] = v
        include_impl = case.get("include_impl") or [impl for impl in essential_impls if impl in case.get("entries", {})]
        variant = case.get("variant", "default")

        # Parameterization: support new schema with defaults + scale
        params = case.get("params", {})
        defaults = params.get("defaults", {})
        scale = params.get("scale", {})
        scale_key = scale.get("key")
        scale_values = list(scale.get("values", [])) if isinstance(scale.get("values", []), list) else []
        env_map = case.get("env_map", {})  # map param->ENV_VAR

        # Compose base env mapping for scripts (Python/Julia). RunMat reads via getenv too.
        case_env = case.get("env", {})
        def build_env_and_assign(assign_overrides: Dict[str, Any]) -> Dict[str, str]:
            merged = {**defaults, **assign_overrides}
            e: Dict[str, str] = {}
            # Map to environment for Python/Julia
            for k, v in merged.items():
                if k in env_map:
                    e[env_map[k]] = str(int(v) if isinstance(v, bool) or isinstance(v, int) else v)
            # Provide a RunMat assignment prelude for reliability (avoids parser issues with top-level if)
            assign_lines = []
            for k, v in merged.items():
                if isinstance(v, bool):
                    assign_lines.append(f"{k} = {1 if v else 0};")
                elif isinstance(v, int):
                    assign_lines.append(f"{k} = {v};")
                elif isinstance(v, float):
                    assign_lines.append(f"{k} = {v};")
            if assign_lines:
                e["HARNESS_ASSIGN"] = "\n".join(assign_lines) + "\n"
            # Merge any case-level env overrides (e.g., device policy)
            for ck, cv in case_env.items():
                e[str(ck)] = str(cv)
            return e

        # Warmup (without recording)
        if warmup:
            env0 = build_env_and_assign({scale_key: scale_values[0] if scale_values else None} if scale_key else {})
            _run_bench(case_dir, 1, per_case_timeout, ROOT / "target" / "suite_warmup.json", variant, overrides, include_impl, env0)

        out_path = ROOT / "benchmarks" / "results" / f"suite_{case_id}.json"
        ensure_output_path(out_path)

        aggregate = {"case": case_dir, "results": [], "sweep": {"param": scale_key, "values": scale_values}}
        for val in (scale_values or [None]):
            env_i = build_env_and_assign({scale_key: val} if (scale_key and val is not None) else {})
            # Run once and collect per-impl entries, annotating x param
            single = _run_bench(case_dir, iterations, per_case_timeout, out_path, variant, overrides, include_impl, env_i)
            for r in single.get("results", []):
                if val is not None:
                    r[scale_key] = val
                aggregate["results"].append(r)
        case_result = aggregate

        # Parity check
        parity_ok = True
        parity_details: List[Dict[str, Any]] = []
        if metric_regex:
            # reference is python-numpy if available
            ref_vals: Dict[int, float] = {}
            for r in case_result.get("results", []):
                if r.get("impl") == "python-numpy":
                    # determine x by checking sweep param or 'n'
                    xkey = scale_key or "n"
                    if xkey in r:
                        val = _extract_metric(r.get("stdout_tail", ""), metric_regex)
                        if val is not None:
                            ref_vals[int(r[xkey])] = val
            for r in case_result.get("results", []):
                impl = r.get("impl")
                xkey = scale_key or "n"
                nval = r.get(xkey)
                if impl == "python-numpy" or nval is None:
                    continue
                ref = ref_vals.get(int(nval)) if ref_vals else None
                got = _extract_metric(r.get("stdout_tail", ""), metric_regex)
                ok = True
                if ref is not None and got is not None:
                    ok = abs(got - ref) <= (atol + rtol * abs(ref))
                else:
                    ok = False
                parity_ok = parity_ok and ok
                parity_details.append({"impl": impl, "n": nval, "ok": ok, "ref": ref, "got": got})

        results["cases"].append(
            {
                "id": case_id,
                "label": label,
                "parity_ok": parity_ok,
                "parity": parity_details,
                "sweep": case_result.get("sweep"),
                "results": case_result.get("results"),
            }
        )

    out = Path(args.output)
    ensure_output_path(out)
    results["suite"]["finished_at"] = datetime.utcnow().isoformat() + "Z"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({"WROTE": str(out)}, indent=2))


if __name__ == "__main__":
    main()
