# Builtin Packaging & Authoring Blueprint

This document captures the builtin authoring, GPU integration, and automation blueprint for the RunMat project.

## Goals
- One Rust source file per builtin containing code, GPU/fusion specs, and unit tests.
- Inventory-backed metadata that fuels both the runtime (Ignition/Turbine + Accelerate) and authoring tools.
- First-class support for scalar and variadic signatures, GPU offload, fusion planning, and BLAS/LAPACK fallbacks.
- Tooling that can emit structured metadata for the Next.js site and drive Codex-based authoring sessions.

## Source Layout
```
crates/runmat-runtime/
  src/
    builtins/
      mod.rs              # category re-exports, shared helpers
      common/             # shared utilities (complex math, GPU helpers, test support)
      math/
        sin.rs
        sum.rs
        ...
      array/
        zeros.rs
        ...
      accel/
        gpu_array.rs
        ...
      ...                 # other categories (io, introspection, strings, etc.)
```
- `builtins/mod.rs` exposes category modules and re-exports existing function symbols to keep downstream code compiling.
- Shared helpers live under `builtins/common/`. They must never perform registration; builtin files call into them explicitly.
- Each builtin file is self-contained: specs, one or more `#[runtime_builtin]` annotated functions, helper routines, and tests.

## Builtin Template Checklist
1. `//!` file doc comment summarising the builtin.
2. `use` statements scoped to required helpers.
3. Optional `pub const GPU_SPEC: BuiltinGpuSpec` and `pub const FUSION_SPEC: BuiltinFusionSpec`, registered via helper macros.
4. One or more `#[runtime_builtin(...)]` functions. Variadic signatures use a trailing `Vec<Value>` parameter, e.g. `rest: Vec<Value>`. The runtime macro already detects this pattern and passes the remaining arguments through.
5. Helper functions (private) to keep the annotated functions concise. Host/GPU split helpers are common.
6. `#[cfg(test)] mod tests` covering: scalars, array/broadcast, variadic combinations, and GPU provider execution (under `feature = "native-accel"`).

## GPU & Fusion Spec Types
- Authoritative implementations of types live in `crates/runmat-runtime/src/builtins/common/spec.rs`. The Function Manager links against the same module to stay in sync.
- Spec constants must carry `#[runmat_macros::register_gpu_spec]` / `#[runmat_macros::register_fusion_spec]` so Accelerate and the Function Manager can discover them (inventory on native targets, generated registry on wasm).
- Provider hooks map to methods exposed by `runmat-accelerate-api`. For reductions, add `ProviderHook::Reduction { name: "reduce_sum" }`.
- `notes` must stay concise (one or two sentences) and focus on actionable implementation details: provider prerequisites, fallbacks, precision caveats, or residency expectations.

### Planner Constants and Fusion
- Builtins should document any constants the planner will inline (e.g. `dim`, `'omitnan'`).
- Use `constant_strategy: InlineLiteral` for small immediates; prefer uniform buffers for runtime switches across kernels.
- Reductions must clearly define output shape for `dim=1` (column-wise) vs `dim=2` (row-wise), and behavior for dims > ndims.

### Two-Pass Reductions
- When `reduce_len` exceeds the workgroup size, providers should use a two-pass kernel:
  - Pass 1: per-slice partial reductions (one workgroup per slice × chunks).
  - Pass 2: reduce partials across chunks.
- Builtins should set `two_pass_threshold` and optionally `workgroup_size` to guide generation.
- NaN handling (`omitnan`/`includenan`) must be honored consistently across passes.

### Testing and Benchmarks
- Provide CPU vs GPU parity tests covering:
  - dim=1 and dim=2; include/omitnan; empty and scalar edge cases.
  - Fused producer → reduction (e.g., `sin(X).*X + 2` → `sum(..., dim)`).
  - Large shapes to validate two-pass speedup; include warmup to amortize pipeline compile.

### FunMatFunc Authoring Hints
- Include a minimal checklist in each builtin for argument parsing, planner constants, GPU spec fields, fusion WGSL body, and examples.

## BLAS / LAPACK Integration Points
- BLAS/LAPACK-backed builtins live under `src/blas.rs` and `src/lapack.rs` guarded by `#[cfg(feature = "blas-lapack")]`.
- When authoring builtins that rely on these crates, mention the feature flag in the GPU spec notes if relevant.
- The Function Manager should check cargo features and warn when attempting to run tests that require BLAS/LAPACK but the feature is disabled.

## RunMat Function Manager Snapshot
- Binary crate at `tools/runmatfunc/`.
- Responsibilities: discover builtin manifests, assemble authoring contexts, launch interactive/headless Codex sessions, run targeted tests, emit documentation bundles, and manage job queues.
- CLI surface (initial):
  - `runmatfunc builtin <name> [--headless] [--model ...]`
  - `runmatfunc browse`
  - `runmatfunc docs emit`
  - `runmatfunc queue add/run`
  - `runmatfunc list`
- Metadata export writes `docs/generated/builtins.json` and `docs/generated/builtins.d.ts` for the Next.js site.

## Accelerate Provider Integration
- Complete the accelerate API and implement the provider hooks for the builtins if needed.
- The provider hooks are located in `crates/runmat-accelerate-api/src/lib.rs`.
- The fallback (simple provider) is located in `crates/runmat-accelerate/src/simple_provider.rs`.
- The WGPU provider is located in `crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs`. 
  - IMPORTANT: MAKE SURE THIS PROVIDER IS FULLY IMPLEMENTED AND TESTED FOR THE BUILTIN.
- Ensure that the provider hooks are implemented for the builtins that need them.
- The goal is to have a fully functional GPU implementation of the builtins, with the ability to run on the GPU if the provider is registered.

## Legacy Module Cleanup
- Legacy modules are located in `crates/runmat-runtime/src/builtins/*.rs` (e.g. not in a subdirectory).
- Once you are finished authoring a module, search the legacy modules for any legacy implementations of the function and remove them, and clean up any references to them in the codebase.

## Note for Tensor Residency
- For creation builtins that may be the start of a fused expression, ensure both the CPU and GPU paths are implemented and tested, allowing the fusion planner to allocate tensors on the GPU if it is profitable to do so from the start. This is particularly important for builtins that create tensors that are used in subsequent operations, such as `eye`, `zeros`, `ones`, `rand`, `randn`, etc.
- If there are hooks and/or changes that need to be made to the Fusion planner to support this, make them.
- Make sure both paths are tested to ensure they produce the same result.
