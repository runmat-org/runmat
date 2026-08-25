# Testing Strategy

RunMat tests are organized by ownership boundary. Parser and lowering crates test language structure, VM tests exercise bytecode and interpreter behavior, runtime tests cover builtins and providers, integration tests cover cross-crate execution behavior, and WASM tests cover browser and JavaScript-hosted behavior.

There are 8,000+ tests in the RunMat codebase, systematically covering the language pipeline, execution engines, runtime builtins, acceleration layer, plotting, CLI, LSP, workspace replay, filesystem, and WASM bindings.

## Baseline Checks

To run the baseline tests, run the following commands:

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo check --all-targets --all-features
RUST_TEST_THREADS=1 cargo test --all-targets --all-features
```

CI serializes tests with `RUST_TEST_THREADS=1` because several suites touch global runtime state: filesystem providers, GPU providers, registries, process environment, and plotting/runtime hooks.

## Crate-Level Tests

Use crate tests when a change is localized. For example:

| Change area | Useful commands |
| --- | --- |
| Lexer/parser syntax | `cargo test -p runmat-lexer`, `cargo test -p runmat-parser` |
| HIR/MIR lowering | `cargo test -p runmat-hir`, `cargo test -p runmat-mir` |
| VM bytecode/interpreter | `cargo test -p runmat-vm`, `cargo test -p runmat-vm --test indexing` |
| Runtime builtins | `cargo test -p runmat-runtime` |
| CLI behavior | `cargo test -p runmat-cli` |
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
| FieldTrip | `2e14f7291090b19568827799096daec7dfb99de8` | `COPYING` | `src/det2x2.c`, `src/nansum.c` |
| hctsa | `f89569f78a2889a410ba9c0100f780120a3d750a` | `LICENSE.txt` | `Toolboxes/Physionet/sampen_mex.c`, `Toolboxes/gpml/util/minfunc/mex/lbfgsAddC.c` |

```bash
RUNMAT_MEX_FIELDTRIP_CHECKOUT=/path/to/fieldtrip \
RUNMAT_MEX_HCTSA_CHECKOUT=/path/to/hctsa \
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
