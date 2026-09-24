# RM-1219 validation — September 10, 2026

The four-language float64 CPU comparison is numerically valid. RunMat's default
GPU pricing pipeline and its repeated-JIT pricing diagnostic have reproducible
failures in the official public runtime. These results are not ready for a public
performance claim; no failed row contributes a timing or speedup.

## Release and machine

The [official latest runtime release](https://github.com/runmat-org/runmat/releases/tag/v0.6.2)
was v0.6.2 when checked. Downloaded its macOS aarch64 asset separately, verified
its SHA-256 against GitHub's asset digest, and ran that binary without replacing
the user's installed CLI. Both binaries report 0.6.2 but have different hashes.

Apple M4, Mac16,13, 24 GiB RAM. GNU Octave 11.3.0, Julia 1.12.7.
CPU float64: Python 3.12.14 / NumPy 2.3.5. CPU float32 and GPU:
Python 3.14 / NumPy 2.5.3 / PyTorch 2.14.0. Julia GPU uses Metal with a pinned
project and manifest. RunMat telemetry confirms Metal kernels on Apple M4.
See [release evidence](evidence/2026-09-10/release.json) and
[environment details](evidence/2026-09-10/environment.txt).

Julia and Octave were installed through Homebrew. PyTorch and its dependencies
were installed in the isolated /tmp/rm1219-python environment. The official
RunMat binary is /tmp/rm1219-tools/runmat. Reproduction instructions are in
[OPTIONS.md](OPTIONS.md); temporary tool directories can be recreated.

## CPU comparison

Three warmups and ten timed repetitions within each process, identical input
fixtures, float64, acceleration disabled. These are warm execution samples,
not proof that the pricing kernel compiled through RunMat's JIT. Fresh process
startup and fixture loading are excluded. NumPy and Julia are faster on this
workload and machine; no claim about all scientific workloads follows.

| Implementation | Paths | Median ms / outcome |
| --- | ---: | --- |
| runmat | 100,000 | 8.700 |
| octave | 100,000 | 1.210 |
| python-numpy | 100,000 | 0.572 |
| julia | 100,000 | 0.331 |
| runmat | 1,000,000 | 105.010 |
| octave | 1,000,000 | 11.392 |
| python-numpy | 1,000,000 | 6.381 |
| julia | 1,000,000 | 3.216 |
| runmat | 10,000,000 | 1084.850 |
| octave | 10,000,000 | 115.846 |
| python-numpy | 10,000,000 | 67.945 |
| julia | 10,000,000 | 28.731 |

[Raw float64 results](evidence/2026-09-10/cpu-float64.json).
All four scalar-loop implementations also pass at 1K and 10K paths;
[raw loop checks](evidence/2026-09-10/cpu-loops.json). Loop results are for
correctness and bounded execution checks, not a JIT speed comparison.

The separate [float32 CPU run](evidence/2026-09-10/cpu-float32.json) passes for
RunMat, NumPy and Julia at all sizes. Octave passes at 100K but misses the
predeclared standard-error tolerance at 1M and 10M. Preserve these failures;
do not silently compare double-precision Octave against single-precision GPUs.

## GPU comparison

All GPU runs use float32-rounded shared fixtures. Timings include pricing,
statistics, synchronization and scalar result retrieval. Resident runs exclude
the initial input upload. Absolute and relative tolerance are both 2e-5 against
a float64 reference evaluated on the same rounded inputs, unchanged for failures.

| Implementation | Paths | Median ms / outcome |
| --- | ---: | --- |
| runmat | 100,000 | FAILED correctness |
| python-torch | 100,000 | 1.000 |
| julia | 100,000 | 1.946 |
| runmat | 1,000,000 | FAILED correctness |
| python-torch | 1,000,000 | 2.065 |
| julia | 1,000,000 | 2.348 |
| runmat | 10,000,000 | FAILED correctness |
| python-torch | 10,000,000 | FAILED correctness |
| julia | 10,000,000 | 11.539 |

[Raw default GPU results](evidence/2026-09-10/gpu-default.json).

- RunMat's price passes, but standard error is wrong at every tested size.
  At 100K paths, reported standard error is 2970.529296875 versus reference
  0.044568205417637514. The incorrect value repeats across all 13 samples.
- Julia Metal passes at 100K, 1M and 10M.
- PyTorch MPS passes at 100K and 1M. At 10M, one sample in the original run
  reports 0.004530449397861958 versus reference 0.004461145711416805; the other
  samples pass. A [short retry](evidence/2026-09-10/pytorch-recheck-short.json)
  and a [full ten-sample retry](evidence/2026-09-10/pytorch-recheck-full.json)
  both pass with the identical fixture hash. Treat the first run as invalid
  and the mismatch as intermittent, not a proven deterministic library defect.

### Separately labeled RunMat diagnostic

`--disable-fused-reduction` makes 100K and 1M pass, but changes the execution
configuration. At 10M it still fails: price 9.389544486999512 versus reference
9.411349262530187, repeated in every sample. This is not a general workaround
and is never substituted into the default result.

| Implementation | Paths | Median ms / outcome |
| --- | ---: | --- |
| runmat | 100,000 | 98.092 |
| runmat | 1,000,000 | 323.880 |
| runmat | 10,000,000 | FAILED correctness |

[Diagnostic raw results](evidence/2026-09-10/gpu-diagnostic.json).

The upload-inclusive run passes for RunMat with this diagnostic flag, PyTorch,
and Julia at 100K and 1M. These times include input upload and scalar output
retrieval, but exclude RNG, file I/O and startup. Apple unified-memory timings
must not be generalized to discrete-GPU transfer costs.

| Implementation | Paths | Median ms / outcome |
| --- | ---: | --- |
| runmat | 100,000 | 110.694 |
| python-torch | 100,000 | 1.405 |
| julia | 100,000 | 2.127 |
| runmat | 1,000,000 | 331.904 |
| python-torch | 1,000,000 | 2.292 |
| julia | 1,000,000 | 3.172 |

[Upload-inclusive raw results](evidence/2026-09-10/gpu-upload-inclusive.json).

## JIT diagnostic

Minimal arithmetic and exp loops each report six JIT and six interpreter
executions across twelve measured runs (after three warmups):
[arithmetic](evidence/2026-09-10/jit-arithmetic.txt),
[exp](evidence/2026-09-10/jit-exp.txt).

The pricing core, with no fixture I/O, string dispatch, timing or printing,
fails at measured iteration 7 with `RunMat:UndefinedFunction` when `--jit` is
set. The same script without `--jit` completes all twelve iterations. This
places the failure near the observed JIT activation boundary, but does not yet
identify which internal call is failing. Do not label the warm CPU timing
results as successful JIT execution.

Repro: [diagnostics/jit_pricing.m](diagnostics/jit_pricing.m).
[Failure output](evidence/2026-09-10/jit-pricing.txt),
[interpreter success](evidence/2026-09-10/interpreter-pricing.txt).

```sh
# cpu.toml contains [runtime.accelerate], enabled=false,
# allow_inprocess_fallback=false.
RUNMAT_CONFIG=/path/to/cpu.toml /path/to/official/runmat benchmark benchmarks/monte-carlo-analysis/diagnostics/jit_pricing.m --jit --iterations 12
RUNMAT_CONFIG=/path/to/cpu.toml /path/to/official/runmat benchmark benchmarks/monte-carlo-analysis/diagnostics/jit_pricing.m --iterations 12
```

GPU uncertainty repro: [diagnostics/gpu_uncertainty.m](diagnostics/gpu_uncertainty.m).
With the hardware provider enabled, default fused reductions report uncertainty
around 2294.188 versus a CPU-only reference around 0.0274313. Disabling fused
reductions makes this smaller deterministic repro pass. Main-suite evidence
shows the workaround fails at larger sizes.

## Status and next work

Six unit tests pass, including rejection of the observed GPU mismatches.
The existing failures are retained in raw JSON. No changes were made to the
runtime, installed RunMat executable, or public website, and nothing is committed
or published. Source hashes in each run identify the local scripts used.

Prioritize the pricing JIT failure and GPU reduction correctness, then repeat
these exact fixtures before expanding performance claims. RNG-inclusive runs,
peak-memory profiling, independent performance repetitions, and website charts
remain outside this validation pass. RM-1219 remains In Progress.
