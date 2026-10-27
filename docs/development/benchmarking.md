# Benchmarking

RunMat has two benchmarking utilities in this repo. The CLI `benchmark` command measures one MATLAB script through the normal session engine. The `benchmarks/` harness compares RunMat against other implementations across representative workloads and records telemetry for performance analysis.

## CLI Benchmark Command

Use `runmat benchmark` when validating a single script or a focused runtime change:

```bash
runmat benchmark script.m --iterations 5 --jit
```

The command resolves the script, creates a session, warms up three times, runs the requested iteration count, and prints:

- total iterations
- JIT execution count
- interpreter execution count
- total time
- average time
- throughput

Use `--jit` when measuring JIT behavior. Without it, the benchmark runs through the default session configuration.

For local raw command-boundary measurements, create a new JSONL report:

```bash
runmat benchmark benchmarks/elementwise-math/runmat.m --jit --warmup 3 --iterations 31 --report elementwise.jsonl
```

The report adds one first invocation in a fresh session, then records the requested warmups and each repeated sample separately. `cold_first` means first invocation of this session, not an empty process-global, provider or disk cache. `warm` labels the repeated phase, not a guarantee that every selected physical recipe is warm. Without `--report`, the existing default three warmups and aggregate output remain; `--warmup` can explicitly change their count. Zero iterations is rejected.

Each raw nanosecond sample times the same driven execution request through command return and existing cleanup. Session construction, frozen-source capture, profile snapshots, executable hashing and report writes are outside that clock. Human aggregate results retain their existing evaluator-millisecond boundary and must not be mixed with the report's raw command-boundary samples.

The identity row hashes the captured source text actually reused by every request and reads the current executable for a SHA256 identity (or reports unavailable). It does not substitute a Git revision or package version for executable bytes. Requested provider/JIT settings are separate from observed execution. The reported JIT threshold records the input setting; native tiering does not use it. Enabling JIT permits native execution but does not force a particular tier. See [JIT Compilation Pipeline](/docs/runtime/jit/pipeline) for tiering and placement policies. There is no benchmark force-specialized, force-GPU or force-fusion option.

Per-sample feedback deltas retain the canonical typed signature, candidate identity and kind. They describe completed feedback, not every selected or declined route; no feedback means `no_observed_feedback`, not an inferred interpreter execution. Eviction/reset/schema changes and counter underflow are explicit unknown history. CPU/OSR/vectorized counter deltas and `used_jit` are supplementary, not replacements for a matching route identity. Different region and whole-call signatures must not be combined into one execution-cost comparison.

Core projects sealed physical/deferred facts at its existing admission-to-seal join. Their canonical coverage remains `incomplete`: pending-free counts do not certify that all possible routes were observed. Provider-wide transfer-byte deltas are supplementary isolated-run diagnostics, not command-attributed completion or device time. Only existing certified profiling supplies optional device time and kernel counts; unknown metrics remain null. A fusion plan or queued dispatch is not proof of successful fused physical execution.

Qualification workloads must independently assert values/precision and include any required `gather`/host-result dependency **inside** the measured source. The command does not insert a gather, copy results for an oracle, change numerical classes, or claim that every returned value is host-resident. Use identical frozen workloads, precision and result boundaries across settings, and retain modes not actually observed as not observed. Report completion alone is not a numerical/performance acceptance verdict or a claim that CPU/GPU routes all ran.

Reports use exclusive creation, stream/flushing records instead of retaining an iteration-sized buffer, and keep partial output on errors. A terminal `failed_or_abandoned` row distinguishes normal failure/unwinding from completion; I/O failure or abrupt process death may leave no terminal row, which is never success. Output files are not overwritten and no telemetry consent is changed.

## Cross-Language Benchmark Suite

The `benchmarks/` directory contains reproducible workload comparisons. Each case has a RunMat script and comparable implementations in Python, PyTorch, Julia, or Octave where applicable.

The suite includes these cases:

| Case | Purpose |
| --- | --- |
| `elementwise-math` | Long elementwise chains that stress fusion and memory bandwidth. |
| `4k-image-processing` | Image-sized array transforms and parity checks. |
| `monte-carlo-analysis` | Monte Carlo GBM-style workloads and stochastic update chains. |

The harness prefers a `runmat` binary on `PATH`. If one is not available, it falls back to `cargo run -q -p runmat --release --`, which is useful but slower because Cargo participates in the run.

Build a release binary before collecting numbers:

```bash
cargo build -p runmat --release --features wgpu
```

Run one case:

```bash
cd benchmarks
python3 ./.harness/run_bench.py \
  --case 4k-image-processing \
  --iterations 3 \
  --output ../results/4k_image_processing.json
```

Run the suite and generate plots:

```bash
cd benchmarks
python3 ./.harness/run_suite.py \
  --suite ./.harness/suite.json \
  --output ../results/suite_results.json

python3 ./.harness/plot_suite.py \
  --input ../results/suite_results.json \
  --output_dir ../results
```

Suite output is JSON. Plot output includes scaling, speedup, and telemetry charts.

## Parity And Metrics

Benchmark cases print `RESULT_ok` markers with case-specific metrics such as MSE or price. The suite config defines the expected regexes and parity checks, then records pass/fail details in the result JSON, allowing the harness to compare numerical results and program outputs between different underlying implementations.

`run_bench.py` records implementation name, language, iteration count, median time, all timings, stdout and stderr tails, optional case metrics, device data, and RunMat provider telemetry. Treat checked-in result files as historical artifacts; regenerate results on the machine and GPU being evaluated.

## GPU Telemetry

The benchmark harness can capture RunMat acceleration telemetry by setting `RUNMAT_TELEMETRY_OUT` and `RUNMAT_TELEMETRY_RESET` for RunMat runs. The CLI also exposes direct acceleration inspection:

```bash
runmat accel-info
runmat accel-info --json --reset
runmat accel-calibrate benchmarks/results/suite_results.json --dry-run --json
runmat accel-calibrate --refresh --dry-run --json
runmat accel-calibrate --refresh --json
```

Use telemetry when deciding whether a benchmark is measuring the intended execution path. For GPU work, verify provider selection, offload decisions, residency behavior, and fallback counts before interpreting raw timings.

Suite residual CPU-time/category allocations are diagnostic estimates, not isolated CPU measurements. The harness marks them with schema 2 and `diagnostic-residual-v1`; importing them with `--dry-run` reports unchanged coefficients and `residual-diagnostic` provenance. They cannot be committed as calibration. `--auto-calibrate` invokes `accel-calibrate --refresh`, which reuses the runtime's existing original CPU templates, provider comparisons, and cleanup owner. `--dry-run` measures without persisting; a committed refresh writes the provider/precision-specific v2 cache. Automatic sample counts are not fabricated: the refresh outcome has `runs: null` because that owner does not export a sample census. Input files and `--refresh` are mutually exclusive.

Qualified original-template imports require schema 2, the closed `original-scan-mac-v1` unit policy, exact provider identity, and `F32`/`F64` precision. Historical scan/MAC denominators are unchanged. Output publication/conversion is separately identified cold-prior work, not a relabeled template measurement. Legacy category profile fitting, `RUNMAT_ACCEL_PROFILE`, and auto-offload `profile_path` do not participate in decisions or calibration. `wgpu_profile` remains a diagnostic benchmark; matched whole-call observations belong to the canonical PlacementSession owner.

GPU submission time measures how long the host takes to enqueue work; it does not measure completed device execution. Fusion placement traces report successful submission as a queue event, without a completion event or a successful execution-cost sample. Measure device completion and any required transfers or synchronization when comparing CPU and GPU latency. A CPU-selected fusion region records completion only after its bytecode reaches the region's normal exit; errors, cancellation, and abandoned execution do not contribute successful timing samples.

On Metal, RunMat uses compute-pass timestamp queries but does not enable encoder-wide timestamp writes, which can prevent submissions from completing. The full-command device interval is therefore unavailable; it is not replaced by the sum of compute-pass times or a host synchronization time. Compare completed end-to-end latency when transfer and synchronization costs matter.

## Performance Smoke Tests

Some performance checks live as Rust tests. They are not a replacement for benchmark runs, but they catch severe regressions in JIT, residency, and GPU dispatch behavior.

```bash
cargo test -p runmat-jit
cargo test -p runmat-core native_tiering
cargo test -p runmat-runtime-integration-tests --test bench_residency_smoke
cargo test -p runmat-accelerate --features wgpu --test matmul_residency
cargo test -p runmat-vm --test fusion_gpu
```

The JIT and Core tiering tests check bounded warmup, publication and cache reuse, exact-profile specialization, loop feedback and OSR, transactional interactive execution, and steady-state compilation counts. Residency tests check that GPU values remain on device across representative operation chains.

## Interactive Preparation and Plot Timing

For interactive script latency, use the runtime's existing `tracing` stream in addition
to the command-boundary benchmark. `runtime.prepare_product` contains source capture,
`runtime.compile` (including parse, lowering, source binding, MIR analysis and registry
publication), draft construction, dependency binding, runtime source binding, registry
projection and artifact admission/publication. `runtime.prepare_workspace` separates
workspace setup and variable materialization from interpretation. These spans use
`skip_all`; source text, paths, tensor contents and workspace values are not recorded.
`runtime.environment` separates linked runtime and builtin catalog fingerprinting;
browser catalog identity must account for registrations submitted after startup.

For animated plots, `plot.surface.render_data` measures surface preparation and
`plot.present` measures web presentation. `runtime.pause.present` identifies rendering
before and after a pause; `runtime.pause.wait` records the requested delay and follows
the cancellable wait using `tracing::Instrument`. Synchronous span guards never cross
an await. Async spans are entered on each poll: their active intervals are not the
elapsed timer delay. Compare the first/last timestamps and intervening idle intervals,
and do not add parent and child durations together.

Browser integrations can collect these through the SDK's existing
`subscribeTraceEvents`/`unsubscribeTraceEvents` API, or inspect forwarded local events
with `otell search 'runtime.prepare_product|runtime.compile|runtime.prepare_workspace|runtime.pause|plot.present|plot.surface.render_data' --service runmat-runtime`.
Compare the same script, precision, project/session state, renderer attachment and
logging level; a fresh in-memory session is a control, not an equivalent app benchmark.

## Direct-call Overhead

The direct-call microbenchmark compares scalar `strcmp` and scalar/array `numel` calls with and without a placement session:

```bash
cargo bench -p runmat-accelerate --bench direct_call_observation --no-default-features -- 10000 31
```

The arguments select repeated-call iterations and sample count. CSV output includes the workload, placement mode, first-call or repeated-call phase, sample number, iteration count, and nanoseconds per call. Inputs and contexts are constructed outside timing; each measured call includes dispatch, execution, result creation, and result release. Results are checked afterward.

First-call samples use fresh sessions, not fresh processes. Repeated-call samples prime their session before timing and continue updating its feedback. The placement/no-placement contrast measures the whole placement-enabled call path, not allocation cost alone. To measure a change's overhead, run the same benchmark on both source revisions with matching compiler, features, settings, and machine conditions. These host-only measurements do not establish CPU/GPU speedups or end-to-end script performance.

## WGPU Profiling

The acceleration crate includes a WGPU profiling binary for backend-level measurements:

```bash
cargo run -p runmat-accelerate --features wgpu --bin wgpu_profile -- --output wgpu_profile.json
cargo run -p runmat-accelerate --features wgpu --bin wgpu_profile -- --quick --reduce-sweep --output wgpu_profile.json
cargo run -p runmat-accelerate --features wgpu --bin wgpu_profile -- --kernel-probe
```

Use this tool for WGPU backend questions that need lower-level measurements than the CLI or benchmark harness provides.
