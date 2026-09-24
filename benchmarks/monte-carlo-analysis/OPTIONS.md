# European option pricing validation suite (RM-1219)

See [VALIDATION.md](VALIDATION.md) for the verified-release comparison, JIT
diagnostics, GPU correctness failures, and links to raw evidence. The historical
initial measurements at the bottom of this file predate that validation.

This extends the existing Monte Carlo case with **exact terminal GBM sampling**.
The existing multi-step GBM scripts remain a separate workload. Their timings
must not be compared directly with this variant.

Parameters: S0 = K = 100, r = 0.03, sigma = 0.2, maturity = 1 year. The analytic
European call value is approximately 9.413403383853016. All implementations use
float64 and calculate the discounted payoff mean and sample standard error.

## Run

Use Python with NumPy installed. Install RunMat, GNU Octave, and Julia for the
complete CPU comparison. The Python executable launching the runner also runs
the NumPy implementation. Pass `--runmat /absolute/path/to/runmat` to select the
exact release binary; otherwise it uses PATH, never an implicit Cargo build.

```sh
python3 benchmarks/.harness/run_options.py --output /tmp/options-cpu.json
```

The default sizes are 100K, 1M, and 10M paths in both loop and vectorized modes,
with three warmups and ten measured repetitions per process. A smaller local run:

```sh
python3 benchmarks/.harness/run_options.py --sizes 1000,100000 --implementations runmat,python-numpy --output /tmp/options-smoke.json
python3 -m unittest discover -s benchmarks/.harness -p test_options.py
```

## What is measured

- The fixture is generated once per size using NumPy default_rng(seed=1219),
  written as little-endian float64, and read unchanged by each language.
  JSON records its SHA-256, NumPy version, source hashes, and runtime versions.
- Timed regions include payoff allocation, pricing, and statistics. Fixture I/O,
  random-number generation, and process startup are excluded from these regions.
- `first_execution_ms` includes first-use costs within the timed region.
  `process_total_ms` includes the complete process, fixture load, warmups, all
  repetitions, and output. It is **not** a single-run latency or startup metric.
- RunMat acceleration is explicitly disabled through a temporary configuration;
  CPU thread limits are recorded. Julia uses fused broadcasting in the
  vectorized version. Python loops use scalar math for payoff generation, then
  NumPy statistics. Loop and vectorized modes are labeled separately.
- Every repetition must match the shared reference price and standard error
  within rtol=atol=1e-10. The fixture mean must also lie within six sample
  standard errors of the analytic price. Missing samples, timeouts, and process
  failures produce failed rows with no speed result. Missing requested runtimes
  are recorded and cause a nonzero exit, rather than silently disappearing.

## Apple GPU comparison

Use Python with NumPy and PyTorch installed. The verified GPU environment used
Python 3.14, NumPy 2.5.3 and PyTorch 2.14.0. Julia's Metal dependencies are pinned
in `julia-gpu/Project.toml` and `Manifest.toml`.

```sh
julia --project=benchmarks/monte-carlo-analysis/julia-gpu -e 'using Pkg; Pkg.instantiate()'
python3 benchmarks/.harness/run_options.py --implementations runmat,python-torch,julia --julia-project benchmarks/monte-carlo-analysis/julia-gpu --modes vectorized --device gpu --precision float32 --output /tmp/options-gpu.json
```

GPU work uses explicit gpuArray, PyTorch MPS, and Julia MtlArray inputs. PyTorch's
MPS fallback and RunMat's provider fallback are disabled. RunMat must also emit
Metal kernel telemetry; that proves GPU activity, not the absence of every
per-operation host fallback. Float32 has a separately declared rtol=atol=2e-5
for price and standard error against a float64 reference evaluated on the same
float32-rounded inputs. This tolerance is unchanged when a result fails.

`--timing resident` uploads inputs before timing, then includes pricing,
statistics, synchronization, and returning the two scalar results to the CPU.
`--timing end-to-end` additionally uploads the input on every repetition. This
means input-to-result timing; it still excludes file I/O, RNG, and startup.
The implementation does not download the full payoff array because the requested
output is the two scalar statistics. Apple unified memory is not equivalent to
a discrete GPU transfer benchmark.

`--disable-fused-reduction` is a RunMat diagnostic only, explicitly recorded in
JSON. It must never be presented as the default RunMat result. The discovered
correctness failures and remaining size limitations are listed in VALIDATION.md.

## Remaining before publication

This is an initial correctness and CPU timing baseline, not a finished public
performance claim. JSON deliberately sets `publication_ready: false`.

- Resolve the reproduced JIT pricing failure and GPU uncertainty failure before
  claiming successful warmed-JIT or default-GPU option-pricing performance.
- Add end-to-end RNG-inclusive runs with statistical checks, separate startup
  measurement, memory profiling, full hardware details, and independent process
  repetitions to assess between-process variance.
- Resolve or explain all failed GPU cases; retain the matched-precision CPU and
  GPU results separately and collect independent process repetitions.
- Add scaling charts and the website detail page after these checks pass.

## Initial local verification — September 10, 2026

RunMat 0.6.2 (installed release binary), Python 3.12.14, NumPy 2.3.5,
macOS, CPU acceleration disabled. Three warmups and ten measured samples in
one process per configuration. These are development observations, not a
current-release or production performance claim.

| Paths | Mode | RunMat median (ms) | Python median (ms) |
| --- | --- | ---: | ---: |
| 1,000 | Loop | 93.689 | 0.176 |
| 1,000 | Vectorized | 0.227 | 0.012 |
| 100,000 | Vectorized | 8.050 | 0.650 |
| 1,000,000 | Vectorized | 96.595 | 6.369 |

All eight measured configurations passed price and standard-error parity.
The 1M-path shared-fixture estimate was 9.394655438813963, within the
predeclared statistical bound around the analytic value. Large scalar-loop
runs were stopped before completion; no large-loop results are claimed.
Julia/Octave ports are implemented but unverified because those runtimes were
not installed. No GPU measurements were collected.

Five unit tests passed. Separate integration probes confirmed that missing
runtimes and nonzero process exits return failure and produce no median timings.
Next investigate the current RunMat release and JIT execution evidence before
interpreting the local performance gap.
