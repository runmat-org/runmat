---
title: "Testing Strategy"
category: "Development"
section: "14.3"
last_updated: "May 28, 2026"
---

# Testing Strategy

RunMat tests are organized by ownership boundary. Parser and lowering crates test language structure, VM tests exercise bytecode and interpreter behavior, runtime tests cover builtins and providers, integration tests cover cross-crate execution behavior, and WASM tests cover browser and JavaScript-hosted behavior.

There are 8,000+ tests in the RunMat codebase, systematically covering the language pipeline, execution engines, runtime builtins, acceleration layer, plotting, CLI, LSP, snapshotting, filesystem, and WASM bindings.

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
RUNMAT_GENERATE_WASM_REGISTRY=1 cargo build -p runmat-wasm --target wasm32-unknown-unknown
scripts/test-wasm-headless.sh
```

`scripts/test-wasm-headless.sh` regenerates the WASM registry, checks `runmat-core` for wasm compatibility, and runs browser-based WASM tests. To include runtime browser tests:

```bash
RUNMAT_WASM_INCLUDE_RUNTIME=1 scripts/test-wasm-headless.sh
```

Focused WASM regression suites are available for symptom and replay coverage:

```bash
scripts/test-wasm-regression-suite.sh symptom-closure
scripts/test-wasm-regression-suite.sh replay-smoke
```

Those wrappers run the appropriate `wasm-pack test --node` and `wasm-pack test --chrome --headless` targets under `crates/runmat-wasm/tests`.

## Macro UI Tests

`runmat-macros` uses compile-fail fixtures for macro diagnostics.

```bash
cargo test -p runmat-macros --test compile
```

The fixtures live under `crates/runmat-macros/tests/ui`. Each failing Rust input has a matching `.stderr` expectation. Update those expectations only when the diagnostic change is intentional.

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