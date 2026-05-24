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
    core.rs
    fft.rs
    helpers.rs
    init.rs
    ops/
      constructors.rs
      elementwise.rs
      image.rs
      indexing.rs
      linalg.rs
      polynomial.rs
      random.rs
      reduction.rs
      signal.rs
      tensor.rs
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

- Final workspace-wide green verification after all in-flight changes are committed.

## Plan-of-Record Decisions

These are mandatory, not optional:

1. Final provider implementation root is `backend/wgpu/provider/*`.
2. Operation semantics live in `backend/wgpu/provider/ops/*`.
3. Lifecycle/infrastructure stays explicit at provider root (`init.rs`, `core.rs`, `helpers.rs`, plus `backend.rs`).
4. Launch plumbing remains in `dispatch/*`; shader source remains in `shaders/*`.
5. Random module naming is normalized (`random.rs`), avoiding long-term `rnd` abbreviation drift.

## Final Target Layout

```text
backend/wgpu/
  mod.rs
  provider.rs
  provider/
    backend.rs
    init.rs
    core.rs
    helpers.rs
    ops/
      elementwise.rs
      reduction.rs
      tensor.rs
      indexing.rs
      random.rs
      constructors.rs
      polynomial.rs
      image.rs
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
  - `WgpuProvider`, `WgpuProviderOptions`, `BufferEntry`, trait impl surface, top-level orchestration.
- `provider/init.rs`:
  - provider initialization/bootstrap/env parsing/autotune+warmup setup.
- `provider/core.rs`:
  - handle table, storage/uniform buffers, submit/readback/upload/download/free, metadata cleanup.
- `provider/helpers.rs`:
  - small cross-family provider utilities (non-semantic helpers only).
- `provider/ops/*`:
  - operation-family semantics and provider-level orchestration.
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

The refactor is in the final stretch. The implementation has already moved off the monolith and into `provider/*`. The remaining plan-of-record work is to finish the explicit `ops/*` layout and normalized random naming, then close out with full workspace-green verification.
