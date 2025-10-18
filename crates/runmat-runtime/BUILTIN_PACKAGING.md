# Builtin Packaging & Authoring Blueprint

This document captures the end-state we want for builtin authoring, GPU integration, documentation, and automation. It is the go-to reference when wiring new builtins or extending the RunMat Function Manager tooling.

## Goals
- One Rust source file per builtin containing code, long-form documentation, GPU/fusion specs, and unit tests.
- Inventory-backed metadata that fuels both the runtime (Ignition/Turbine + Accelerate) and authoring tools.
- First-class support for scalar and variadic signatures, GPU offload, fusion planning, and BLAS/LAPACK fallbacks.
- Tooling that can emit structured docs for the Next.js site and drive Codex-based authoring sessions.

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
- Each builtin file is self-contained: documentation constant, specs, one or more `#[runtime_builtin]` annotated functions, helper routines, and tests.

## Builtin Template Checklist
1. `//!` file doc comment summarising the builtin.
2. `use` statements scoped to required helpers.
3. `pub const DOC_MD: &str = r#"..."#;` containing YAML frontmatter + Markdown (details below).
4. Optional `pub const GPU_SPEC: BuiltinGpuSpec` and `pub const FUSION_SPEC: BuiltinFusionSpec`, registered via helper macros.
5. One or more `#[runtime_builtin(..., doc_md = DOC_MD, ...)]` functions. Variadic signatures use a trailing `Vec<Value>` parameter, e.g. `rest: Vec<Value>`. The runtime macro already detects this pattern and passes the remaining arguments through.
6. Helper functions (private) to keep the annotated functions concise. Host/GPU split helpers are common.
7. `#[cfg(test)] mod tests` covering: scalars, array/broadcast, variadic combinations, GPU provider execution (under `feature = "native-accel"`), and doc example smoke tests via the shared test harness.

## Inline Documentation Expectations
- YAML frontmatter should cover `title`, `category`, `keywords`, `summary`, `references`, `gpu_support`, `fusion`, `tested`, and any flags relevant to BLAS/LAPACK usage.
- Markdown body should explain numerics, broadcasting, error behaviour, GPU semantics (including how Accelerate fuses kernels and manages residency), and thorough examples of usage within the MATLAB language syntax. Aim for 5-10 examples, and pick examples based on the most common use cases that would be searched on a search engine for when using the function.
- Encourage users to understand GPU offload: describe gpuArray creation, gather, and the lazy execution model (Ignition + Accelerate detect fusion opportunities, queue kernels, and execute on demand).

## GPU & Fusion Spec Types
- Authoritative implementations of types live in `crates/runmat-runtime/src/builtins/common/spec.rs`. The Function Manager links against the same module to stay in sync.
- `register_builtin_*` macros submit the specs into inventory so Accelerate and the Function Manager can discover them.
- Provider hooks map to methods exposed by `runmat-accelerate-api`. For reductions, add `ProviderHook::Reduction { name: "reduce_sum" }`.
- `notes` must stay concise (one or two sentences) and focus on actionable implementation details: provider prerequisites, fallbacks, precision caveats, or residency expectations. Avoid repeating long-form documentation that already exists in `DOC_MD`.

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
- Keep `DOC_MD` examples executable; prefer small, deterministic inputs for CI.

## BLAS / LAPACK Integration Points
- BLAS/LAPACK-backed builtins live under `src/blas.rs` and `src/lapack.rs` guarded by `#[cfg(feature = "blas-lapack")]`.
- When authoring builtins that rely on these crates, mention the feature flag in `DOC_MD` (`requires_feature: blas-lapack`) and in the GPU spec notes if relevant.
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
- Documentation export writes `docs/generated/builtins.json` and `docs/generated/builtins.d.ts` for the Next.js site.

## Legacy Module Cleanup
- Legacy modules are located in `crates/runmat-runtime/src/builtins/legacy/`.