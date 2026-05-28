# RunMat Accelerate WGPU Provider Refactor

## Objective

Refactor the WGPU provider implementation into a clear module tree with strict domain boundaries, while preserving behavior exactly:

- no public API changes
- no fallback policy changes
- no kernel math/parameter changes
- no submission ordering/readback behavior changes
- no host/device result changes

This remains a logic-preserving cleanup.

## Repo-Verified Current State (2026-05-25)

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
    core.rs
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
      linalg/
        decomposition.rs
      linalg.rs
      polynomial.rs
      random.rs
      reduction.rs
      signal.rs
      solve.rs
      telemetry.rs
      tensor.rs
      window.rs
  dispatch/*
  shaders/*
```

What is already complete:

- Monolith extraction by operation family is complete.
- Lifecycle split is complete (`init/core/helpers`).
- Canonical package rename from `provider_impl/*` to `provider/*` is complete.
- Public facade remains `backend/wgpu/provider.rs`.

Verification status:

- `cargo check --workspace --all-targets` is green.
- `cargo test --workspace --all-targets` is green.
- `cargo clippy --workspace --all-targets -- -D warnings` is green.

## Plan-of-Record Decisions

These are mandatory, not optional:

1. Final provider implementation root is `backend/wgpu/provider/*`.
2. Operation semantics live in `backend/wgpu/provider/ops/*`.
3. Lifecycle/infrastructure stays explicit at provider root (`init.rs`, `core.rs`, `helpers.rs`, plus `backend.rs`).
4. Backend shape/support types live in dedicated files (`backend_types.rs`, `backend_shared.rs`), not mixed into operation semantics.
5. Trait method surface is centralized in `provider/trait_impl.rs` and remains dispatch-only.
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
      linalg/
        decomposition.rs
      linalg.rs
      signal.rs
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
- `provider/trait_impl.rs`:
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

### Phase 10: Post-Refactor Quality Pass

1. Continue reducing trait-surface usage inside `ops/*` (prefer internal `*_exec` APIs).
2. Keep splitting oversized operation modules (`ops/linalg.rs`, `ops/reduction.rs`) into focused submodules.
3. Deduplicate repeated numerical pipelines where behavior is identical.

## Verification Requirements

After each structural phase:

- build/test relevant crate slices
- verify no behavioral deltas through existing tests
- avoid any semantic edits unless explicitly tracked as separate fixes

## Bottom Line

The refactor structure is now in place with dispatch centralized in `trait_impl.rs`, operation logic extracted into `ops/*`, and full workspace validation green. Remaining work is quality/maintainability refinement, not structural migration.
