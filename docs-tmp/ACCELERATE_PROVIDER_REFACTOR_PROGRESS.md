# Accelerate Provider Refactor Progress

## 2026-05-24

- Started execution against `docs-tmp/ACCELERATE_PROVIDER_REFACTOR.md`.
- Phase 0: normalized WGPU provider implementation module file layout from `backend/wgpu/provider_impl.rs` to `backend/wgpu/provider_impl/mod.rs`.
- Scope intent: structural move only; no behavior change.
- Verification for Phase 0:
  - `cargo test -p runmat-accelerate --test provider_init` passed.
  - `cargo check -p runmat-accelerate` encountered a pre-existing workspace warning-as-error in `runmat-runtime` (`ImpulseResponse.discrete` dead-code), not introduced by this refactor move.
- Plan doc updated to reflect current repository state and final plan-of-record wording:
  - canonical monolith path references now point to `provider_impl/mod.rs`
  - sequence declared mandatory (no optional phases)
  - Phase 0 explicitly marked complete
  - boundary language tightened for `ops/image.rs`, `dispatch/*`, and `core` ownership
- Phase 1a: moved logical/comparison WGSL constants out of `provider_impl/mod.rs` into `backend/wgpu/shaders/logical.rs`, then imported from provider implementation.
- Verification for Phase 1a:
  - `cargo test -p runmat-accelerate --test provider_init` passed.
  - `cargo check -p runmat-accelerate` still blocked by the same pre-existing workspace warning-as-error in `runmat-runtime` (`ImpulseResponse.discrete` dead-code).
- Phase 1b: extracted contiguous elementwise provider methods into `backend/wgpu/provider_impl/elementwise.rs` and wired `mod elementwise;` in `provider_impl/mod.rs`.
  - moved methods: `elem_{eq,ne,lt,le,gt,ge}_exec`, `logical_*_exec`, `unary_op_exec`, `scalar_op_exec`.
  - no kernel parameter or dispatch logic changes; code moved verbatim.
- Verification for Phase 1b:
  - `cargo test -p runmat-accelerate --test provider_init` passed.
  - `cargo test -p runmat-accelerate --lib` passed.
- Phase 1c: moved binary execution helpers into `provider_impl/elementwise.rs`.
  - moved methods: `binary_op_exec`, `binary_op_broadcast_exec`.
  - kept the call graph unchanged (`dot_exec`, `cross_exec`, and trait entrypoints still call these helpers exactly as before).
- Verification for Phase 1c:
  - `cargo test -p runmat-accelerate --test provider_init` passed.
  - `cargo test -p runmat-accelerate --lib` passed.
- Phase 1d: moved fused elementwise kernel executors into `provider_impl/elementwise.rs`.
  - moved methods: `fused_elementwise_exec`, `fused_elementwise_multi_exec`.
  - reduction methods remain in `provider_impl/mod.rs` for the Phase 2 extraction.
- Verification for Phase 1d:
  - `cargo test -p runmat-accelerate --test provider_init` passed.
  - `cargo test -p runmat-accelerate --lib` passed.
