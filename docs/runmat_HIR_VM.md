# RunMat boundary migration summary

## Reason: boundary leakage in the pipeline

- The interpreter/JIT stack currently crosses into the runtime for almost every primitive operation (`crates/runmat-ignition/src/vm.rs:181-2000`), so letting `runmat_accelerate_api::provider()` escape into builtins scattered across the tree makes telemetry, gatekeeping, and fallback behavior hard to keep consistent.
- GPU slicing/indexing also invoked the acceleration API directly, which duplicated provider counter updates and warning emissions inside many builtin implementations (`crates/runmat-ignition/src/vm.rs:5921-6566`, `crates/runmat-runtime/src/accel_provider.rs`), exposing seams that leak into the interpreter boundary when the runtime falls back to host tensors.
- These leaks defeat the notion of a living boundary; the helper must now shield every builtin and test from directly importing the provider so we keep the shared trace counter, warning strings, and error telemetry under one roof.

## Chosen approach

### Code-level mission (individual fixes)

- Keep the VM/runtime boundary healthy by having the interpreter always call `runmat_runtime::call_builtin`, which now routes GPU work through `accel_provider::maybe_provider` so the runtime trace counter and telemetry stay centralized (`crates/runmat-runtime/src/dispatcher.rs:23-70` and `crates/runmat-runtime/src/elementwise.rs:1-280`).
- Guard the few GPU-only tests that still import `crate::accel_provider` with `#[cfg(feature = "wgpu")]` to avoid unused-import lints while keeping the helper accessible when wgpu is enabled (e.g., `crates/runmat-runtime/src/builtins/array/shape/ipermute.rs:8-14`).
- Document the boundary crossings that still exist (workspace resolvers, interaction handlers, GC barriers) so reviewers see exactly which seams remain after this migration.

### Builtin mission (systematic macro work)

- Extend `runmat_macros::runtime_builtin` so every generated catalog entry includes an `accel_provider` flag/guard setter; the macro can then import `runmat_runtime::accel_provider::{maybe_provider, maybe_provider_for_handle, provider_for_handle}` and emit the telemetry-friendly guard without touching the implementation body.
- Rebuild the builtin inventory (`tools/builtin_inventory/`) once the macro is extended so the generated list of helpers reflects the new contracts instead of referencing raw `runmat_accelerate_api::provider()` calls scattered across `crates/runmat-runtime/src/builtins`.
- Repeatedly rerun the conversion script (`tools/convert_accel_provider.py`) whenever a builtin or test still imports `runmat_accelerate_api::provider()` to keep the helper-based pattern in sync.

## Tools & documentation

- PyComby is the structural search/replace tool referenced for these migrations; the updated README at `https://github.com/bardo84/pycomby` now presents a generic "way forward" (pattern matching → helper injection → verify workflow) that matches our macro-driven approach without naming RunMat specifics.
- The living document content above is the PR-ready story: reason, code approach, builtin approach, tooling, and next steps, so reviewers can point to `docs/runmat_HIR_VM.md` and see why we did the migration and how.

## Validation

- `cargo test -p runmat-runtime workspace` (ran with an extended timeout after the first invocation hit the 124 s limit) confirms that the helper changes compile and the workspace/introspection tests still pass.
- `tests/functions/closure_resolver_script.m` became the regression guard for nested closures and script-defined handles, ensuring the shared resolver still serves HIR bodies when the JIT and interpreter cross the workspace boundary.

## Next steps before closing the migration

1. Extend the macro's `accel_provider` metadata and rebuild the builtin inventory so new helpers automatically get the shared guard.
2. Run the async/stdin and workspace test harnesses (`cargo test -p runmat-core async_stdin -- --test-threads=2`, `cargo test -p runmat-runtime workspace`) after each macro/builtin change to verify TLS handler isolation remains stable.
3. Push the updated docs/code once the macro extension and builtin rebuild are complete, referencing this summary in the PR description so reviewers understand the boundary story.
