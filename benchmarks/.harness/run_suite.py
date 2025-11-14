#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Local imports
try:
    from .utils import ensure_output_path, which
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from utils import ensure_output_path, which  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
def _prebuild_runmat() -> None:
    bin_path = ROOT / "target" / "release" / "runmat"
    if bin_path.exists():
        return
    # Allow opt-out
    if os.environ.get("SKIP_PREBUILD_RUNMAT"):
        return
    cmd = [
        "cargo",
        "build",
        "-p",
        "runmat",
        "--release",
        "--features",
        "wgpu",
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


def _ns_to_ms(value: Optional[int]) -> float:
    if value is None:
        return 0.0
    return float(value) / 1_000_000.0


def _summarize_speedups(results: List[Dict[str, Any]], x_param: Optional[str]) -> Optional[Dict[str, Any]]:
    baseline_key = x_param or "n"
    baseline = {}
    for entry in results:
        if entry.get("impl") != "python-numpy":
            continue
        param_val = entry.get(baseline_key)
        median_ms = entry.get("median_ms")
        if param_val is None or median_ms in (None, 0):
            continue
        try:
            baseline[int(param_val)] = float(median_ms)
        except Exception:
            continue
    if not baseline:
        return None

    series = []
    impl_names = sorted({entry.get("impl") for entry in results if entry.get("impl") and entry.get("impl") != "python-numpy"})
    for impl in impl_names:
        pts: List[Dict[str, Any]] = []
        for entry in results:
            if entry.get("impl") != impl:
                continue
            param_val = entry.get(baseline_key)
            median_ms = entry.get("median_ms")
            if param_val is None or median_ms in (None, 0):
                continue
            try:
                key = int(param_val)
            except Exception:
                continue
            base_ms = baseline.get(key)
            if not base_ms:
                continue
            speedup = base_ms / float(median_ms)
            pts.append({"param": key, "speedup": speedup})
        if not pts:
            continue
        pts.sort(key=lambda v: v["param"])
        speeds = [p["speedup"] for p in pts]
        mean_speed = sum(speeds) / len(speeds)
        auc_val = 0.0
        if len(pts) >= 2:
            xs = [float(p["param"]) for p in pts]
            ys = speeds
            for i in range(len(xs) - 1):
                width = xs[i + 1] - xs[i]
                if width <= 0.0:
                    continue
                auc_val += width * (ys[i] + ys[i + 1]) / 2.0
        series_entry: Dict[str, Any] = {"impl": impl, "points": pts, "mean_speedup": mean_speed}
        if auc_val > 0.0:
            series_entry["auc_trapz"] = auc_val
        series.append(series_entry)

    if not series:
        return None
    return {"baseline": "python-numpy", "x_param": baseline_key, "series": series}


def _summarize_runmat_telemetry(results: List[Dict[str, Any]], x_param: Optional[str]) -> Optional[Dict[str, Any]]:
    if not results:
        return None
    per_point: List[Dict[str, Any]] = []
    device_info: Optional[Dict[str, Any]] = None
    total_upload = 0
    total_download = 0
    total_fused_elem_count = 0
    total_fused_elem_ms = 0.0
    total_fused_red_count = 0
    total_fused_red_ms = 0.0
    total_matmul_count = 0
    total_matmul_ms = 0.0
    total_fusion_hits = 0
    total_fusion_misses = 0
    total_bind_hits = 0
    total_bind_misses = 0

    telemetry_error_counts: Dict[str, int] = defaultdict(int)
    missing_device_runs = 0
    missing_telemetry_runs = 0

    calib_elem_time = 0.0
    calib_elem_units = 0.0
    calib_red_time = 0.0
    calib_red_units = 0.0
    calib_matmul_time = 0.0
    calib_matmul_units = 0.0
    calibration_runs = 0
    calibration_provider: Optional[Dict[str, Any]] = None    

    auto_thresholds: Optional[Dict[str, Any]] = None
    auto_base_source: Optional[str] = None
    auto_env_override = False
    auto_calibrate_ms: Optional[int] = None
    auto_reason_counts: Dict[str, int] = defaultdict(int)
    auto_decision_counts: Dict[str, int] = defaultdict(int)

    for entry in results:
        telemetry_payload = entry.get("provider_telemetry") or {}
        if not telemetry_payload:
            continue

        device_entry = telemetry_payload.get("device")
        if device_entry:
            if device_info is None:
                device_info = device_entry
        else:
            missing_device_runs += 1

        error_msg = telemetry_payload.get("error")
        telemetry_raw = telemetry_payload.get("telemetry")
        if telemetry_raw is None:
            missing_telemetry_runs += 1
        telemetry = telemetry_raw or {}
        fused_elem = telemetry.get("fused_elementwise") or {}
        fused_red = telemetry.get("fused_reduction") or {}
        matmul = telemetry.get("matmul") or {}
        upload_bytes = int(telemetry.get("upload_bytes") or 0)
        download_bytes = int(telemetry.get("download_bytes") or 0)
        fusion_hits = int(telemetry.get("fusion_cache_hits") or 0)
        fusion_misses = int(telemetry.get("fusion_cache_misses") or 0)
        bind_hits = int(telemetry.get("bind_group_cache_hits") or 0)
        bind_misses = int(telemetry.get("bind_group_cache_misses") or 0)

        fe_count = int(fused_elem.get("count") or 0)
        fe_ms = _ns_to_ms(fused_elem.get("total_wall_time_ns"))
        fr_count = int(fused_red.get("count") or 0)
        fr_ms = _ns_to_ms(fused_red.get("total_wall_time_ns"))
        mm_count = int(matmul.get("count") or 0)
        mm_ms = _ns_to_ms(matmul.get("total_wall_time_ns"))

        total_upload += upload_bytes
        total_download += download_bytes
        total_fused_elem_count += fe_count
        total_fused_elem_ms += fe_ms
        total_fused_red_count += fr_count
        total_fused_red_ms += fr_ms
        total_matmul_count += mm_count
        total_matmul_ms += mm_ms
        total_fusion_hits += fusion_hits
        total_fusion_misses += fusion_misses
        total_bind_hits += bind_hits
        total_bind_misses += bind_misses

        median_raw = entry.get("median_ms")
        median_ms = float(median_raw) if isinstance(median_raw, (int, float)) else 0.0

        status = "ok"
        point_entry: Dict[str, Any] = {
            "median_ms": median_raw,
            "upload_bytes": upload_bytes,
            "download_bytes": download_bytes,
            "fused_elementwise": {"count": fe_count, "wall_ms": fe_ms},
            "fused_reduction": {"count": fr_count, "wall_ms": fr_ms},
            "matmul": {"count": mm_count, "wall_ms": mm_ms},
            "fusion_cache": {"hits": fusion_hits, "misses": fusion_misses},
            "bind_group_cache": {"hits": bind_hits, "misses": bind_misses},
        }
        if error_msg:
            msg = str(error_msg)
            telemetry_error_counts[msg] += 1
            status = "error"
            point_entry["error"] = msg
        elif telemetry_raw is None:
            status = "missing"
        point_entry["status"] = status

        param_key = x_param or "n"
        param_val = entry.get(param_key)
        if param_val is not None:
            point_entry["param"] = {param_key: param_val}

        auto = telemetry_payload.get("auto_offload") or {}
        if auto:
            if auto_thresholds is None and auto.get("thresholds"):
                auto_thresholds = auto.get("thresholds")
            if auto_base_source is None and auto.get("base_source"):
                auto_base_source = auto.get("base_source")
            if auto.get("env_overrides_applied"):
                auto_env_override = True
            if auto_calibrate_ms is None and auto.get("calibrate_duration_ms") is not None:
                auto_calibrate_ms = auto.get("calibrate_duration_ms")
            decisions = auto.get("decisions") or []
            if decisions:
                reason_counts: Dict[str, int] = defaultdict(int)
                disposition_counts: Dict[str, int] = defaultdict(int)
                cpu_decisions: List[Dict[str, Any]] = []
                for d in decisions:
                    reason = d.get("reason")
                    if reason:
                        auto_reason_counts[str(reason)] += 1
                        reason_counts[str(reason)] += 1
                    disposition = d.get("decision")
                    if disposition:
                        auto_decision_counts[str(disposition)] += 1
                        disposition_counts[str(disposition)] += 1
                    if str(disposition).lower() == "cpu":
                        cpu_decisions.append(d)
                point_entry["auto_offload"] = {
                    "decisions_total": len(decisions),
                    "by_reason": dict(reason_counts),
                    "by_decision": dict(disposition_counts),
                }

                if cpu_decisions:
                    cat_units = {"elementwise": 0.0, "reduction": 0.0, "matmul": 0.0}
                    for d in cpu_decisions:
                        op = str(d.get("operation") or "").lower()
                        if op in ("elementwise", "unary", "transpose"):
                            elems = d.get("elements")
                            if elems is not None:
                                try:
                                    cat_units["elementwise"] += float(elems)
                                except (TypeError, ValueError):
                                    pass
                        elif op == "reduction":
                            elems = d.get("elements")
                            if elems is not None:
                                try:
                                    cat_units["reduction"] += float(elems)
                                except (TypeError, ValueError):
                                    pass
                        elif op == "matmul":
                            flops = d.get("flops")
                            if flops is not None:
                                try:
                                    cat_units["matmul"] += float(flops)
                                except (TypeError, ValueError):
                                    pass

                    total_units = sum(cat_units.values())
                    if total_units > 0.0:
                        cpu_time_ms = max(0.0, median_ms - (fe_ms + fr_ms + mm_ms))
                        if cpu_time_ms > 0.0:
                            calibration_runs += 1
                            if calibration_provider is None and device_info is not None:
                                calibration_provider = dict(device_info)
                            for cat, units in cat_units.items():
                                if units <= 0.0:
                                    continue
                                share = cpu_time_ms * (units / total_units)
                                if cat == "elementwise":
                                    calib_elem_time += share
                                    calib_elem_units += units
                                elif cat == "reduction":
                                    calib_red_time += share
                                    calib_red_units += units
                                elif cat == "matmul":
                                    calib_matmul_time += share
                                    calib_matmul_units += units

        per_point.append(point_entry)

    if not per_point:
        return None

    summary: Dict[str, Any] = {
        "runs": len(per_point),
        "per_point": per_point,
        "totals": {
            "upload_bytes": total_upload,
            "download_bytes": total_download,
            "fused_elementwise": {
                "count": total_fused_elem_count,
                "wall_ms": total_fused_elem_ms,
            },
            "fused_reduction": {
                "count": total_fused_red_count,
                "wall_ms": total_fused_red_ms,
            },
            "matmul": {
                "count": total_matmul_count,
                "wall_ms": total_matmul_ms,
            },
            "fusion_cache": {
                "hits": total_fusion_hits,
                "misses": total_fusion_misses,
            },
            "bind_group_cache": {
                "hits": total_bind_hits,
                "misses": total_bind_misses,
            },
        },
    }

    if device_info:
        summary["device"] = device_info

    if auto_thresholds or auto_reason_counts:
        summary["auto_offload"] = {
            "base_source": auto_base_source,
            "thresholds": auto_thresholds,
            "env_overrides_applied": auto_env_override,
            "calibrate_duration_ms": auto_calibrate_ms,
            "decision_counts_by_reason": dict(auto_reason_counts),
            "decision_counts_by_disposition": dict(auto_decision_counts),
        }

    if calibration_runs:
        calib_payload: Dict[str, Any] = {
            "runs": calibration_runs,
            "cpu_time_ms": {
                "elementwise": calib_elem_time,
                "reduction": calib_red_time,
                "matmul": calib_matmul_time,
            },
            "units": {
                "elementwise": calib_elem_units,
                "reduction": calib_red_units,
                "matmul_flops": calib_matmul_units,
            },
        }
        if calibration_provider is not None:
            calib_payload["provider"] = calibration_provider
        summary["auto_offload_calibration"] = calib_payload

    warnings: List[str] = []
    if telemetry_error_counts:
        issues = ", ".join(
            f"{msg} (runs={count})" for msg, count in telemetry_error_counts.items()
        )
        warnings.append(f"provider telemetry errors detected: {issues}")
    if missing_telemetry_runs and not telemetry_error_counts:
        warnings.append(
            f"provider telemetry payload missing in {missing_telemetry_runs} run(s)"
        )
    if device_info is None and missing_device_runs:
        warnings.append(
            "no GPU device information recorded; verify WGPU-enabled build is in use"
        )
    if warnings:
        summary["warnings"] = warnings

    return summary


def _case_summary(case_result: Dict[str, Any], x_param: Optional[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    results = case_result.get("results", [])
    speedups = _summarize_speedups(results, x_param)
    if speedups:
        summary["speedups"] = speedups
    runmat_results = [r for r in results if r.get("impl") == "runmat"]
    telemetry_summary = _summarize_runmat_telemetry(runmat_results, x_param)
    if telemetry_summary:
        summary["telemetry"] = telemetry_summary
    return summary


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

def _accumulate_calibration(target: Dict[str, Any], sample: Dict[str, Any]) -> None:
    runs = int(sample.get("runs") or 0)
    if runs <= 0:
        return
    target["runs"] = target.get("runs", 0) + runs

    times = sample.get("cpu_time_ms") or {}
    units = sample.get("units") or {}

    target.setdefault("cpu_time_ms", {"elementwise": 0.0, "reduction": 0.0, "matmul": 0.0})
    target.setdefault("units", {"elementwise": 0.0, "reduction": 0.0, "matmul_flops": 0.0})

    target["cpu_time_ms"]["elementwise"] += float(times.get("elementwise") or 0.0)
    target["cpu_time_ms"]["reduction"] += float(times.get("reduction") or 0.0)
    target["cpu_time_ms"]["matmul"] += float(times.get("matmul") or 0.0)

    target["units"]["elementwise"] += float(units.get("elementwise") or 0.0)
    target["units"]["reduction"] += float(units.get("reduction") or 0.0)
    target["units"]["matmul_flops"] += float(units.get("matmul_flops") or 0.0)

    provider = sample.get("provider")
    if provider:
        existing = target.get("provider")
        if existing is None:
            target["provider"] = provider
        elif existing != provider:
            target["provider_conflict"] = True
            target["provider"] = None

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

    suite_calibration: Dict[str, Any] = {
        "runs": 0,
        "cpu_time_ms": {"elementwise": 0.0, "reduction": 0.0, "matmul": 0.0},
        "units": {"elementwise": 0.0, "reduction": 0.0, "matmul_flops": 0.0},
        "provider": None,
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
                            ref_vals[str(r[xkey])] = val
            for r in case_result.get("results", []):
                impl = r.get("impl")
                xkey = scale_key or "n"
                nval = r.get(xkey)
                if impl == "python-numpy" or nval is None:
                    continue
                ref = ref_vals.get(str(nval)) if ref_vals else None
                got = _extract_metric(r.get("stdout_tail", ""), metric_regex)
                ok = True
                if ref is not None and got is not None:
                    ok = abs(got - ref) <= (atol + rtol * abs(ref))
                else:
                    ok = False
                parity_ok = parity_ok and ok
                parity_details.append({"impl": impl, "n": nval, "ok": ok, "ref": ref, "got": got})

        case_entry: Dict[str, Any] = {
            "id": case_id,
            "label": label,
            "parity_ok": parity_ok,
            "parity": parity_details,
            "sweep": case_result.get("sweep"),
            "results": case_result.get("results"),
        }
        summary_payload = _case_summary(case_result, scale_key)
        if summary_payload:
            case_entry["summary"] = summary_payload
            telemetry_summary = summary_payload.get("telemetry") or {}
            calibration_sample = telemetry_summary.get("auto_offload_calibration")
            if calibration_sample:
                _accumulate_calibration(suite_calibration, calibration_sample)            
        results["cases"].append(case_entry)

    out = Path(args.output)
    ensure_output_path(out)
    results["suite"]["finished_at"] = datetime.utcnow().isoformat() + "Z"
    if suite_calibration.get("runs", 0) > 0:
        suite_calib_payload: Dict[str, Any] = {
            "runs": suite_calibration["runs"],
            "cpu_time_ms": suite_calibration["cpu_time_ms"],
            "units": suite_calibration["units"],
        }
        provider_info = suite_calibration.get("provider")
        if provider_info:
            suite_calib_payload["provider"] = provider_info
        if suite_calibration.get("provider_conflict"):
            suite_calib_payload["provider_conflict"] = True
        results["suite"]["auto_offload_calibration"] = suite_calib_payload    
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({"WROTE": str(out)}, indent=2))


if __name__ == "__main__":
    main()
