---
title: "Development"
category: "Development"
section: "14.0"
last_updated: "May 28, 2026"
---

# Development

RunMat is developed as a Rust workspace with a small TypeScript package for the WebAssembly and browser-facing API. Most daily work is done with normal Cargo commands, but the workspace has enough native, GPU, and WASM surface area that developers need to know which command exercises which part of the system.

The development section covers the repository mechanics around building, target support, testing, and performance work. Runtime design details stay in the earlier compiler, VM, GPU, WASM, filesystem, and session sections.

The root `Cargo.toml` owns the crate list and shared dependency versions. Internal crates are version-pinned together, so workspace changes normally build against one coherent RunMat version. The TypeScript package under `bindings/ts` builds the WASM runtime, LSP bundle, generated builtin metadata, and startup snapshot used by JavaScript consumers.

## Page Map

| Page | Purpose |
| --- | --- |
| [Build System](/docs/runtime/development/build-system) | Workspace layout, feature flags, native dependencies, WASM package builds, and release-oriented build commands. |
| [Supported Architectures](/docs/runtime/development/supported-architectures) | Native release triples, WASM target, GPU backend constraints, and platform-specific runtime differences. |
| [Testing Strategy](/docs/runtime/development/testing) | What the CI baseline runs, where tests live, and which focused suites to run for common changes. |
| [Benchmarking](/docs/runtime/development/benchmarking) | CLI benchmarks, cross-language benchmark harnesses, GPU telemetry, and performance smoke tests. |
| [Telemetry](/docs/runtime/development/telemetry) | Runtime analytics envelope, consent, delivery, local provider telemetry, and opt-out behavior. |

## Local Baseline

The Rust toolchain is pinned in `rust-toolchain.toml`. Before sending a broad runtime change through review, mirror the CI baseline:

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo check --all-targets --all-features
RUST_TEST_THREADS=1 cargo test --all-targets --all-features
```

For narrow changes, run the focused crate or integration test first, then expand to the baseline when the change touches shared runtime behavior.
