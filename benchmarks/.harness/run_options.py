"""RM-1219 CPU baseline: fail-closed correctness and in-process samples.

Uses the existing case discovery, but does not use run_impl: that function times
whole processes and currently does not reject nonzero exit codes.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from run_bench import ROOT, build_impl_commands

SAMPLE = re.compile(r"SAMPLE rep=(\d+) ms=(\S+) price=(\S+) stderr=(\S+)")


def analytic_price():
    cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return 100 * cdf(0.25) - 100 * math.exp(-0.03) * cdf(0.05)


def validate_samples(stdout, count, price, stderr, tolerance=1e-10):
    samples = []
    for match in SAMPLE.finditer(stdout):
        rep, ms, p, se = map(float, match.groups())
        if not all(map(math.isfinite, (rep, ms, p, se))) or ms <= 0 or se < 0:
            raise ValueError("nonfinite or invalid sample")
        if not math.isclose(p, price, rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError(f"price parity failed: {p} vs {price}")
        if not math.isclose(se, stderr, rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError(f"standard-error parity failed: {se} vs {stderr}")
        samples.append(dict(rep=int(rep), ms=ms, price=p, stderr=se))
    if [s['rep'] for s in samples] != list(range(1, count + 1)):
        raise ValueError("missing, duplicate, or out-of-order samples")
    return samples


def capture(cmd):
    return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sizes', default='100000,1000000,10000000')
    ap.add_argument('--implementations', default='runmat,octave,julia,python-numpy')
    ap.add_argument('--modes', default='loop,vectorized')
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--warmups', type=int, default=3)
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--runmat', default=shutil.which('runmat'))
    ap.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    ap.add_argument('--precision', choices=['float64', 'float32'], default='float64')
    ap.add_argument('--timing', choices=['resident', 'end-to-end'], default='resident')
    ap.add_argument('--julia-project', type=Path)
    ap.add_argument('--disable-fused-reduction', action='store_true',
                    help='Diagnostic RunMat configuration, never the default GPU result')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sizes = [int(n) for n in args.sizes.split(',')]
    requested = args.implementations.split(',')
    modes = args.modes.split(',')
    if min(sizes) < 2 or args.repeats < 1 or args.warmups < 1 or args.timeout < 1:
        ap.error('sizes >= 2, repeats/warmups/timeout >= 1 required')
    if set(modes) - {'loop', 'vectorized'}:
        ap.error('modes must be loop and/or vectorized')
    if set(requested) - {'runmat', 'octave', 'julia', 'python-numpy', 'python-torch'}:
        ap.error('unknown implementation')
    if args.device == 'gpu' and (args.precision != 'float32' or modes != ['vectorized']):
        ap.error('Apple GPU comparisons require float32 and --modes vectorized')
    if args.device == 'gpu' and set(requested) & {'octave', 'python-numpy'}:
        ap.error('Octave/NumPy are CPU baselines; run them separately')
    if args.precision == 'float32' and 'loop' in modes:
        ap.error('float32 currently supports vectorized mode only; scalar Python math uses float64')
    # Avoid silently preferring an old target/release executable.
    available = [i for i in requested if i != 'runmat' or args.runmat]
    impls = build_impl_commands('monte-carlo-analysis', 'options', [i for i in available if i != 'runmat'])
    if 'runmat' in available:
        impls.insert(0, dict(name='runmat', lang='matlab-syntax', cmd=[
            str(Path(args.runmat).resolve()),
            str(ROOT / 'benchmarks/monte-carlo-analysis/runmat_options.m')]))
    for impl in impls:
        if impl['name'] == 'runmat':
            impl['cmd'][0] = str(Path(args.runmat).resolve())
        if impl['name'] == 'python-numpy':
            impl['cmd'][0] = sys.executable
        if impl['name'] == 'python-torch':
            impl['cmd'][0] = sys.executable
        if impl['name'] == 'julia':
            impl['cmd'].insert(1, '--startup-file=no')
            if args.julia_project:
                impl['cmd'].insert(1, '--project=' + str(args.julia_project.resolve()))
    result = dict(case='monte-carlo-analysis', variant='options-terminal-cpu-fixture',
                  platform=platform.platform(), machine=platform.machine(),
                  cpu=platform.processor(), numpy=np.__version__,
                  commit=capture(['git', 'rev-parse', 'HEAD']),
                  dirty=bool(capture(['git', 'status', '--porcelain'])),
                  precision=args.precision, device=args.device, seed=1219, warmups=args.warmups,
                  repeats=args.repeats, analytic_price=analytic_price(),
                  timing=args.timing, timing_excludes='fixture I/O, RNG, process startup',
                  peak_memory=None, publication_ready=False, results=[],
                  missing=[i for i in requested if i not in {x['name'] for x in impls}])
    source_paths = [Path(__file__), ROOT / 'benchmarks/.harness/run_bench.py']
    source_paths.extend(Path(i['cmd'][-1]) for i in impls)
    result['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in source_paths}
    result['binary_sha256'] = {i['name']: hashlib.sha256(Path(shutil.which(i['cmd'][0]) or i['cmd'][0]).read_bytes()).hexdigest()
                               for i in impls}
    if platform.system() == 'Darwin':
        result['hardware'] = {k: capture(['/usr/sbin/sysctl', '-n', k]) for k in
                              ('hw.model', 'hw.memsize', 'machdep.cpu.brand_string')}
    if args.julia_project:
        result['julia_environment_sha256'] = {
            name: hashlib.sha256((args.julia_project / name).read_bytes()).hexdigest()
            for name in ('Project.toml', 'Manifest.toml') if (args.julia_project / name).exists()}
    with tempfile.TemporaryDirectory(prefix='runmat-options-') as tmp:
        config = Path(tmp) / 'runmat.toml'
        config.write_text('[runtime.accelerate]\nenabled = ' + ('true' if args.device == 'gpu' else 'false') +
                          '\nprovider = "wgpu"\nallow_inprocess_fallback = false\n')
        env = os.environ.copy()
        env.update(RUNMAT_CONFIG=str(config), RUNMAT_ACCEL_AUTO_OFFLOAD='0',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   JULIA_NUM_THREADS='1', MC_REPEATS=str(args.repeats),
                   MC_WARMUPS=str(args.warmups))
        env.update(MC_DEVICE=args.device, MC_PRECISION=args.precision, MC_TIMING=args.timing,
                   PYTORCH_ENABLE_MPS_FALLBACK='0')
        if args.disable_fused_reduction:
            env['RUNMAT_DISABLE_FUSED_REDUCTION'] = '1'
        else:
            env.pop('RUNMAT_DISABLE_FUSED_REDUCTION', None)
        result['runmat_disable_fused_reduction'] = args.disable_fused_reduction
        result['runmat_config'] = config.read_text()
        result['thread_limits'] = {k: env[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                                      'MKL_NUM_THREADS', 'JULIA_NUM_THREADS')}
        for n in sizes:
            z = np.random.default_rng(1219).standard_normal(n)
            if args.precision == 'float32':
                z = z.astype(np.float32).astype(np.float64)
            fixture = Path(tmp) / 'normal.f64'
            z.astype('<f8').tofile(fixture)
            payoff = math.exp(-0.03) * np.maximum(100*np.exp(0.01+0.2*z)-100, 0)
            price, stderr = float(payoff.mean()), float(payoff.std(ddof=1)/math.sqrt(n))
            if abs(price-analytic_price()) > 6*stderr:
                raise ValueError('fixture fails predeclared six-standard-error analytic check')
            env['MC_FIXTURE'] = str(fixture)
            fixture_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
            for impl in impls:
                for mode in modes:
                    env['MC_MODE'] = mode
                    row = dict(impl=impl['name'], mode=mode, paths=n,
                               command=impl['cmd'], fixture_sha256=fixture_hash,
                               reference_price=price, reference_stderr=stderr)
                    telemetry_path = Path(tmp) / 'telemetry.json'
                    telemetry_path.unlink(missing_ok=True)
                    env['RUNMAT_TELEMETRY_OUT'] = str(telemetry_path)
                    env['RUNMAT_TELEMETRY_RESET'] = '1'
                    start = time.perf_counter()
                    try:
                        proc = subprocess.run(impl['cmd'], env=env, text=True,
                                              capture_output=True, timeout=args.timeout)
                        row.update(process_total_ms=(time.perf_counter()-start)*1000,
                                   stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
                        if proc.returncode:
                            raise ValueError(f'process exited {proc.returncode}')
                        if telemetry_path.exists() and impl['name'] == 'runmat':
                            row['provider_telemetry'] = json.loads(telemetry_path.read_text())
                        samples = validate_samples(proc.stdout, args.warmups+args.repeats, price, stderr,
                                                   2e-5 if args.precision == 'float32' else 1e-10)
                        if args.device == 'gpu' and 'DEVICE gpu' not in proc.stdout:
                            raise ValueError('GPU implementation did not confirm device execution')
                        if args.device == 'gpu' and impl['name'] == 'runmat':
                            telemetry = row.get('provider_telemetry', {})
                            if telemetry.get('device', {}).get('backend') != 'metal' or not telemetry.get('telemetry', {}).get('kernel_launches'):
                                raise ValueError('missing RunMat Metal kernel evidence')
                        times = [s['ms'] for s in samples[args.warmups:]]
                        med = statistics.median(times)
                        row.update(status='passed', samples=samples, first_execution_ms=samples[0]['ms'],
                                   median_ms=med, min_ms=min(times), max_ms=max(times),
                                   paths_per_second=n*1000/med)
                    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                        row.update(status='failed', error=str(exc))
                    result['results'].append(row)
                    print(f"{impl['name']} {mode} n={n}: {row['status']}", flush=True)
        result['versions'] = {}
        for impl in impls:
            try:
                result['versions'][impl['name']] = capture([impl['cmd'][0], '--version'])
            except (OSError, subprocess.CalledProcessError) as exc:
                result['versions'][impl['name']] = str(exc)
        if 'python-torch' in requested:
            try:
                result['torch_version'] = capture([sys.executable, '-c', 'import torch; print(torch.__version__)'])
            except (OSError, subprocess.CalledProcessError) as exc:
                result['torch_version'] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if result['missing'] or any(r['status'] != 'passed' for r in result['results']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
