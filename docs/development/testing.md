# Testing Strategy

RunMat tests are organized by ownership boundary. Parser and lowering crates test language structure, VM tests exercise bytecode and interpreter behavior, runtime tests cover builtins and providers, integration tests cover cross-crate execution behavior, and WASM tests cover browser and JavaScript-hosted behavior.

## Baseline Checks

Run the workspace tests through the package groups:

```bash
cargo fmt -- --check
python3 scripts/development/run-workspace-test-group.py --check
RUST_TEST_THREADS=1 python3 scripts/development/run-workspace-test-group.py language
RUST_TEST_THREADS=1 python3 scripts/development/run-workspace-test-group.py runtime
RUST_TEST_THREADS=1 python3 scripts/development/run-workspace-test-group.py geometry-analysis
RUST_TEST_THREADS=1 python3 scripts/development/run-workspace-test-group.py tooling-distributed
```

The groups assign each workspace package once, exercise the supported Clarabel feature profiles, and run harness-free benchmark smoke tests with bounded arguments. Long scientific cohorts use the separate receipt policy in `qualification/scientific/policy.json`. Add `--include-extended` to execute those cohorts directly.

CI serializes tests with `RUST_TEST_THREADS=1` because several suites touch global runtime state: filesystem providers, GPU providers, registries, process environment, and plotting/runtime hooks. The group runner sets `RUST_MIN_STACK=67108864` to match the native CLI command and Tokio-worker stack reserve. Unoptimized native continuations can exceed libtest's default 2-MiB thread stack. The operating system commits pages as they are used; the reserve does not allocate a second workspace. Use the same environment variable when invoking affected native tests directly.

Run compiler and lint checks with the same supported package feature profiles. A workspace-wide `--all-features` command enables mutually exclusive external BLAS and language bindings in the vendored Clarabel solver.

## Crate-Level Tests

Use crate tests when a change is localized. For example:

| Change area | Useful commands |
| --- | --- |
| Lexer/parser syntax | `cargo test -p runmat-lexer`, `cargo test -p runmat-parser` |
| HIR/MIR lowering | `cargo test -p runmat-hir`, `cargo test -p runmat-mir` |
| VM bytecode/interpreter | `cargo test -p runmat-vm`, `cargo test -p runmat-vm --test indexing` |
| Runtime builtins | `cargo test -p runmat-runtime` |
| CLI behavior | `cargo test -p runmat` |
| Config loading | `cargo test -p runmat-config` |
| Builtin registry | `cargo test -p runmat-builtins` |
| Macro expansion | `cargo test -p runmat-macros` |

VM tests use shared helpers that parse MATLAB source, lower through HIR and MIR, compile bytecode, and interpret the result. Those helpers also run tests on a larger stack for deep compile or interpreter cases.

Runtime tests include helper utilities for provider setup, GPU provider locking, filesystem wrappers, and gather operations. Prefer those helpers over open-coded global setup in new tests.

## Runtime Integration Tests

Cross-crate runtime behavior lives in `crates/runmat-runtime-integration-tests`. This crate is intentionally separate from unit tests because it exercises dispatch, GPU handle behavior, dtype behavior, provider wiring, and residency assumptions across multiple crates.

```bash
cargo test -p runmat-runtime-integration-tests
```

Run this suite when a change affects builtin dispatch, GPU values, provider registration, dtype conversion, or execution behavior that crosses crate boundaries.

## GPU Tests

The acceleration crate has focused tests for fusion, provider initialization, residency, reductions, matmul paths, precision boundaries, telemetry, and backend behavior.

```bash
cargo test -p runmat-accelerate
cargo test -p runmat-accelerate --features wgpu
cargo test -p runmat-accelerate --features wgpu --test provider_init
```

Runtime GPU tests use either the deterministic in-process provider or WGPU behind the `wgpu` feature:

```bash
cargo test -p runmat-runtime --features wgpu
cargo test -p runmat-runtime-integration-tests --test bench_residency_smoke
```

Use the in-process provider for semantic tests that should not depend on a physical GPU. Use WGPU tests when validating backend initialization, residency, shader dispatch, or device behavior.

Production WGSL generators also have GPU-free Rust tests. These parse the emitted source with the Naga version paired with WGPU, validate its declared capabilities, and check that removing a required capability rejects the shader:

```bash
cargo test -p runmat-accelerate --lib --no-default-features --features wgpu validation_tests
cargo test -p runmat-accelerate --lib --no-default-features --features wgpu atan2_selected_tests
cargo test -p runmat-accelerate --lib --no-default-features --features wgpu selected_native_power_reduction_uses_the_ordinary_helper_for_edge_controls
```

Add generator cases beside their owning shader or provider module, using the shared validation helper rather than copying shader source into a test. Cover precision, operation, operand layout and specialization variants that alter emitted code. Parsing and validation do not establish device support, numerical accuracy, residency or performance; those require separate execution tests on the relevant backend and precision.

Plotting has a GPU-free source census covering its compute precision variants, workgroup specializations, and vertex shaders through the same source materializer used by production pipelines:

```bash
cargo test -p runmat-plot --no-default-features --lib plotting_shader_sources_parse_and_validate_for_every_declared_variant
```

Run the surface packer's device geometry check explicitly on a machine with a GPU:

```bash
RUNMAT_PLOT_FORCE_GPU_TESTS=1 cargo test -p runmat-plot --no-default-features --lib gpu_surface_layout_matches_non_square_cpu_surface_contract
```

This command requires an adapter and device; their absence fails the test. Do not also set `RUNMAT_PLOT_SKIP_GPU_TESTS`. A normal library run without the force flag does not establish surface GPU execution coverage.

## WASM Tests

WASM CI builds the runtime for `wasm32-unknown-unknown`, then runs the headless browser script:

```bash
rustup target add wasm32-unknown-unknown
scripts/regenerate-wasm-registry.sh
cargo build -p runmat-wasm --target wasm32-unknown-unknown --features occt-wasm-host
scripts/test-wasm-headless.sh
```

`scripts/test-wasm-headless.sh` regenerates the WASM registry with the atomic production `plot-web,occt-wasm-host` flow, checks `runmat-core` for wasm compatibility with the same OCCT host feature enabled, and runs browser-based WASM tests. To include runtime browser tests:

```bash
RUNMAT_WASM_INCLUDE_RUNTIME=1 scripts/test-wasm-headless.sh
```

Focused WASM regression suites are available through the runtime dispatcher:

```bash
scripts/runtime/test-wasm-regression-suite.sh symptom-closure
scripts/runtime/test-wasm-regression-suite.sh replay-smoke
```

Those wrappers run the appropriate `wasm-pack test --node` and `wasm-pack test --chrome --headless` targets under `crates/runmat-wasm/tests`.
They also regenerate the WASM builtin registry before running tests, so focused
regressions exercise the same builtin catalog as packaged browser builds.

The symptom-closure suite includes the shared signal compatibility harness in
`crates/runmat-runtime/tests/fixtures/signal_compatibility_harness.m`. The same
fixture is also run by the CLI integration test
`test_signal_compatibility_harness_cli`, covering CSV import, MAT-file
save/load, FFT magnitude/indexing, filter/conv, and signal window functions
through both host and JavaScript filesystem providers.

## FEA Script Tests

The geometry/analysis CI group excludes the `pipeline::transient::rod_qualification` scientific family, which contains long continuum-refinement and multi-block cold-history fixtures. These tests retain their numerical assertions and are available as a separate qualification lane:

```bash
RUST_MIN_STACK=67108864 cargo test --locked -p runmat-analysis-fea --lib -- \
  --test-threads=1 pipeline::transient::rod_qualification::
```

Allow a dedicated machine and an extended time budget for that lane. An ordinary CI pass does not establish these refinement envelopes. To include the lane in the complete geometry/analysis group, use `python3 scripts/development/run-workspace-test-group.py geometry-analysis --include-extended`. Direct `cargo test` retains its usual selection; ignored diagnostics and benchmarks remain opt-in.

FEA governance, calibration, reporting, and artifact scripts use stdlib `unittest` tests:

```bash
scripts/test-fea-scripts.sh
```

## Macro UI Tests

`runmat-macros` uses compile-fail fixtures for macro diagnostics.

```bash
cargo test -p runmat-macros --test compile
```

The fixtures live under `crates/runmat-macros/tests/ui`. Each failing Rust input has a matching `.stderr` expectation. Update those expectations only when the diagnostic change is intentional.

## C MEX Repository Cohorts

The `runmat-mex` suite includes source fixtures maintained in this repository and an opt-in gate for pinned external projects. The external checkouts are not downloaded during ordinary Cargo tests. Supply local checkouts at the exact revisions below, then run the ignored cohort tests explicitly.

| Project | Revision | License file | Executed gateways |
| --- | --- | --- | --- |
| FieldTrip | `2e14f7291090b19568827799096daec7dfb99de8` | `COPYING` | `src/det2x2.c`, `src/getpid.c` |

```bash
RUNMAT_MEX_FIELDTRIP_CHECKOUT=/path/to/fieldtrip \
cargo test -p runmat-mex --test repository_cohorts -- --ignored
```

Each test verifies the checkout's Git revision and license file before compiling. It then loads and executes the gateway through `runmat-mex` and checks its returned value. This gate is intended for supported-platform qualification and for changes to the SDK, compiler plan, loader, value conversion, or gateway lifecycle.

Generated C-to-Fortran gateways are tracked separately because a successful link requires their native Fortran libraries. The pinned source audits use fmm2d revision `550dae5b77b1e006c8ffae37fc832f8c2b536871` and fmm3dbie revision `ec79660cf39ef048741a26e2b31960fccefa4e27`; full build and execution belongs to the advanced Fortran MEX qualification gate.

## Choosing What To Run

| Change | Start with | Expand to |
| --- | --- | --- |
| Parser or syntax | Touched parser test file | `cargo test -p runmat-parser` |
| Lowering or bytecode | Touched HIR/MIR/VM test | `cargo test -p runmat-vm` |
| Builtin implementation | `cargo test -p runmat-runtime` | Runtime integration tests |
| GPU or fusion | `cargo test -p runmat-accelerate --features wgpu` | Runtime GPU integration tests |
| CLI command | `cargo test -p runmat-cli` | Full workspace tests |
| WASM or TypeScript API | `scripts/test-wasm-headless.sh` | WASM regression suite and `npm test` in `bindings/ts` |
| Macro changes | `cargo test -p runmat-macros --test compile` | Full macro crate tests |

Before merging a broad runtime change, run the full baseline or let CI confirm it on the target runners.
