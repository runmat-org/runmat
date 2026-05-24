# RunMat Accelerate WGPU Provider Refactor

## Objective

Refactor the WGPU provider implementation into a clear module tree with strict domain boundaries, while preserving behavior exactly:

- no public API changes
- no fallback policy changes
- no kernel math/parameter changes
- no submission ordering/readback behavior changes
- no host/device result changes

This remains a logic-preserving cleanup.

## Repo-Verified Current State (2026-05-24)

Current WGPU layout:

```text
backend/wgpu/
  mod.rs
  provider.rs
  provider/
    backend.rs
    backend_shared.rs
    backend_types.rs
    trait_impl.rs
    trait_impl_methods/
      context_constructors_random_poly.rs
      elementwise_tensor_signal.rs
      indexing_io_telemetry.rs
      linalg_advanced_pagefun.rs
      linalg_reduction_core.rs
    core.rs
    fft.rs
    helpers.rs
    init.rs
    ops/
      context.rs
      constructors.rs
      elementwise.rs
      fft.rs
      image.rs
      indexing.rs
      io.rs
      linalg.rs
      polynomial.rs
      random.rs
      reduction.rs
      signal.rs
      solve.rs
      telemetry.rs
      tensor.rs
      window.rs
    solve.rs
    window.rs
  dispatch/*
  shaders/*
```

What is already complete:

- Monolith extraction by operation family is complete.
- Lifecycle split is complete (`init/core/helpers`).
- Canonical package rename from `provider_impl/*` to `provider/*` is complete.
- Public facade remains `backend/wgpu/provider.rs`.

What is still pending:

- Full workspace verification pass and closeout artifacts.

## Plan-of-Record Decisions

These are mandatory, not optional:

1. Final provider implementation root is `backend/wgpu/provider/*`.
2. Operation semantics live in `backend/wgpu/provider/ops/*`.
3. Lifecycle/infrastructure stays explicit at provider root (`init.rs`, `core.rs`, `helpers.rs`, plus `backend.rs`).
4. Backend shape/support types live in dedicated files (`backend_types.rs`, `backend_shared.rs`), not mixed into operation semantics.
5. Trait method surface is split by domain (`trait_impl_methods/*`) and remains dispatch-only.
6. Launch plumbing remains in `dispatch/*`; shader source remains in `shaders/*`.
7. Random module naming is normalized (`random.rs`), avoiding long-term `rnd` abbreviation drift.

## Final Target Layout

```text
backend/wgpu/
  mod.rs
  provider.rs
  provider/
    backend.rs
    backend_shared.rs
    backend_types.rs
    trait_impl.rs
    trait_impl_methods/
      context_constructors_random_poly.rs
      elementwise_tensor_signal.rs
      indexing_io_telemetry.rs
      linalg_advanced_pagefun.rs
      linalg_reduction_core.rs
    init.rs
    core.rs
    helpers.rs
    ops/
      context.rs
      elementwise.rs
      reduction.rs
      tensor.rs
      indexing.rs
      io.rs
      telemetry.rs
      random.rs
      constructors.rs
      polynomial.rs
      image.rs
      linalg.rs
      signal.rs
      fft.rs
      solve.rs
      window.rs
    fft.rs
    solve.rs
    window.rs
```

Notes:

- `fft.rs`, `solve.rs`, `window.rs` remain focused non-ops modules.
- `signal.rs` is part of operation semantics and should sit under `ops/` in the final resting state.

## Domain Boundaries

- `provider/backend.rs`:
  - `WgpuProvider`, `WgpuProviderOptions`, top-level orchestration, module wiring.
- `provider/backend_types.rs`:
  - backend data carriers and operation-facing request/response structs.
- `provider/backend_shared.rs`:
  - backend-wide constants/helpers that are not operation-family semantics.
- `provider/trait_impl.rs` + `provider/trait_impl_methods/*`:
  - trait surface organization and thin dispatch into domain exec methods.
- `provider/init.rs`:
  - provider initialization/bootstrap/env parsing/autotune+warmup setup.
- `provider/core.rs`:
  - handle table, storage/uniform buffers, submit/readback/upload/download/free, metadata cleanup.
- `provider/helpers.rs`:
  - small cross-family provider utilities (non-semantic helpers only).
- `provider/ops/*`:
  - operation-family semantics and provider-level orchestration (`*_exec` methods).
- `dispatch/*`:
  - launch mechanics/bindings/workgroup dispatch plumbing.
- `shaders/*`:
  - shader constants/source only.

## Remaining Execution Plan

### Phase 9: Verification + Closeout

1. Run accelerate-focused verification:
   - `cargo test -p runmat-accelerate --lib`
   - `cargo test -p runmat-accelerate --tests`
2. Run full workspace verification:
   - `cargo test --workspace --all-targets`
3. Resolve any non-refactor incidental failures blocking green.
4. Update progress log with final commit hashes and verification results.

## Verification Requirements

After each structural phase:

- build/test relevant crate slices
- verify no behavioral deltas through existing tests
- avoid any semantic edits unless explicitly tracked as separate fixes

## Bottom Line

The refactor structure is now in place with dispatch-only trait slices and operation logic extracted into `ops/*`. Remaining plan-of-record work is verification and closeout evidence.
