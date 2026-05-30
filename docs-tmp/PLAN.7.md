# Plan 7: Accelerate, Fusion, Residency, And Lifetime Hardening

## Objective

Move acceleration and fusion planning onto the semantic HIR, MIR, and analysis fact model while keeping concrete runtime placement and provider state out of semantic facts.

Plans 0-6 create semantic HIR, MIR, type/shape/effect/execution facts, VM layout, async execution boundaries, complete MATLAB core semantics, project composition, workspace export, and runtime class/builtin metadata. Plan 7 makes accelerate/fusion/GC behavior consume those stable compiler products instead of reconstructing its own compiler model from bytecode and runtime side state.

## Desired Resting State

Acceleration planning uses semantic and MIR facts for eligibility and safety:

- `TypeFact`
- `ShapeFact`
- tensor element-domain facts
- `EffectSummary`
- `WorkspaceEffect`
- `EnvironmentEffect`
- `AsyncBehaviorFact`
- `SpawnSafetyFact`
- `FusibilityFact`
- `ParallelSafetyFact`
- `AccelEligibilityFact`
- `DataMovementPolicyHint`

Runtime/provider layers own concrete placement:

- provider availability
- device choice
- promotion/download
- materialization
- concrete `GpuTensorHandle`
- buffer pooling and reuse
- precision support
- logical tensor storage split across one or more provider buffers
- memory pressure decisions

Fusion candidate construction is driven primarily by semantic HIR plus MIR plus `AnalysisStore`, not bytecode reconstruction.

## Core Invariants

- Semantic analysis may say an expression is fusible, acceleration-eligible, acceleration-preferred, or blocked.
- Semantic analysis must not claim concrete residency on a specific runtime device.
- Runtime planner decides actual placement and materialization.
- Providers own buffer allocation, pooling, pipeline caches, and dispatch details.
- Moving a value between host and device does not change its language type.
- Provider storage layout does not change language shape, element domain, or sparsity facts.
- Complex tensor storage must preserve language element counts and complex-to-real operation summaries.
- Async provider dispatch is represented as async behavior, not hidden runtime magic.
- Provider/runtime handles expose concurrency semantics for spawn safety: immutable sharing, copy-on-write, synchronized mutation, or rejection.
- GC and deterministic release must account for provider-owned resources without leaking provider internals into source semantics.

## Primary Crates

- `runmat-accelerate`
- `runmat-accelerate-api`
- `runmat-vm`
- `runmat-gc`
- `runmat-gc-api`

## Secondary Crates

- `runmat-hir`
- `runmat-mir`
- `runmat-static-analysis`
- `runmat-runtime`
- `runmat-core`
- `runmat-async`

## Implementation Plan

1. Harden execution facts.

Refine:

- `FusibilityFact`
- `ParallelSafetyFact`
- `AccelEligibilityFact`
- `DataMovementPolicyHint`
- `AsyncBehaviorFact`
- `SpawnSafetyFact`
- tensor element-domain facts
- `WorkspaceEffect`
- `EnvironmentEffect`
- materialization hints if needed

2. Define semantic/MIR fusion candidate extraction.

Build fusion candidates from:

- resolved builtin/operator identities
- MIR dataflow regions
- type and shape facts
- tensor element-domain facts
- effect barriers
- workspace-effect barriers
- environment-effect barriers
- async boundaries
- spawn-safety barriers
- binding read/write sets
- loop/control-flow boundaries
- workspace/export boundaries

3. Replace bytecode-only fusion graph construction as the primary planner.

The current bytecode graph can remain as a compatibility or execution-lowering artifact temporarily, but semantic/MIR fusion groups should become the planning source of truth.

4. Preserve runtime placement decisions.

Runtime decides provider availability, input-size thresholds, existing device values, promotion/download, precision support, memory pressure, logical-tensor-to-buffer mapping, and fallback.

5. Replace transitional fusion residency tracking.

Current global buffer-ID tracking should move toward a clearer runtime/provider residency registry if still needed.

6. Clarify `Value::GpuTensor` semantics.

Document and enforce:

- `Value::GpuTensor` means concrete runtime device data
- it does not imply a different language type
- it does not change tensor shape, element domain, sparsity, or requested-output behavior
- handle metadata such as precision/storage/logical flags are runtime/provider facts
- a single logical tensor may be backed by multiple provider buffers when required by backend limits
- user-visible `gpuArray` semantics, if exposed, should be modeled separately as language/runtime class behavior

Complex tensor operations must align runtime values with analysis summaries. Operations such as `abs`, `real`, `imag`, and `angle` produce real tensors with the same language shape; `conj` preserves complex domain.

7. Harden materialization boundaries.

Define when intermediates must materialize:

- workspace export
- observable side effects
- workspace/environment effects such as `load`, `clear`, dynamic eval, globals, persistents, path mutation, cwd mutation, and resolver-cache mutation
- async host boundaries
- unknown calls
- aliasing barriers
- host-only builtin calls
- debugging/inspection hooks
- persistent/global/module storage where policy requires

8. Integrate GC and deterministic release.

Clarify ownership of provider resources, GC tracing/retention, provider pools, value drops, downloads, reassignment, workspace/session clear, and host reset.

This must include spawned task lifetimes. Provider handles captured by spawned futures must be retained safely, released on completion/cancellation/drop, and protected from unsynchronized mutation races.

9. Feed accelerate from runtime class metadata.

Plan 6 metadata can provide method/builtin effects, requested-output behavior, tensor element-domain summaries, shape summaries, async behavior, workspace/environment effects, provider-backed implementation availability, dispatch hooks, and docs/diagnostics.

10. Add diagnostics and telemetry.

Diagnostics should distinguish semantic ineligibility, provider unavailable, runtime size below threshold, unsupported dtype/shape, async/provider fallback, memory pressure, and provider errors.

## Tests

Add tests for semantic/MIR fusion eligibility, effect barriers, workspace/environment-effect barriers, async barriers, spawn-safety barriers, unknown calls, shape instability, real/complex tensor element-domain summaries, runtime provider fallback, host/device movement preserving language type, workspace export materialization, clear/reassignment-driven resource release, residency clearing, provider-internal pooling, multi-buffer backing for large logical tensors where supported, spawned task provider-handle retention/release, and GC/drop behavior.

## Acceptance Criteria

- Fusion candidates are primarily built from semantic HIR plus MIR plus analysis facts.
- Runtime/provider placement decisions remain runtime/provider responsibilities.
- Concrete device residency is not encoded in `TypeFact` or `ShapeFact`.
- Provider layout does not alter language shape or element-domain facts.
- Current bytecode-only fusion reconstruction is removed or reduced to a compatibility/execution-lowering layer.
- Residency bookkeeping is explicit and tied to runtime handles, value replacement, download, release, and GC behavior.
- Provider handles captured by spawned tasks have explicit concurrency and lifetime semantics.
- Accelerate/fusion diagnostics explain semantic vs runtime/provider fallback reasons.

## Explicit Non-Goals

- Do not make provider availability part of semantic correctness.
- Do not make static analysis machine-dependent.
- Do not expose provider buffer identities to user-level language semantics.
- Do not require every operation to have an accelerated implementation.
