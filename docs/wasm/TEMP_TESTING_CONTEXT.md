# Runtime WASM Testing Context (Temporary)

This scratchpad keeps us honest while we unblock `runmat-runtime` tests on
`wasm32-unknown-unknown` + headless Chrome. Update it as each blocker is
triaged, fixed, or deferred.

## Guardrails / Heuristics

- **Test surface**: Every change must keep
  - native `cargo test -- --test-threads=1` (workspace root or crate-level) green,
  - `RUNMAT_GENERATE_WASM_REGISTRY=1 cargo check -p runmat-runtime --target wasm32-unknown-unknown`,
  - `scripts/test-wasm-headless.sh` (default core run), and
  - `RUNMAT_WASM_INCLUDE_RUNTIME=1 scripts/test-wasm-headless.sh` once runtime blockers are addressed.
- **Single-threaded host tests**: Always run host-side `cargo test … -- --test-threads=1`
  to match CI and avoid races in global builtins / filesystem fixtures.
- **Gating strategy**: Prefer `cfg(target_arch = "wasm32")` shims or helper modules over deleting tests.
  Only skip a test when its dependency is fundamentally host-only (Accelerate/OpenBLAS, tokio networking, etc.).
- **Performance bias**: Never slow down native/GPU hot paths just to make wasm happy.
  Use cfgs or alternate code paths so the wasm accommodations stay isolated.
- **Regression safety**: Each blocker fix needs
  - notes here (what broke, how we fixed it, remaining TODOs),
  - automated coverage (existing tests now runnable in wasm, or new wasm-only tests),
  - no API breakage unless documented + coordinated.

## Known Blockers (to keep updated)

1. BLAS/LAPACK tests (`tests/blas_lapack.rs`) require native Accelerate/OpenBLAS. ✅ gated to host-only (`cfg(all(feature="blas-lapack", not(target_arch="wasm32")))`).
2. Plotting ops used `#[ctor]`, which `wasm-bindgen-test` rejects. ✅ tests now call `ensure_plot_test_env()` (per-ops helper) backed by a shared `Once` in `plotting::tests`.
3. Doc example helpers (`test_support`) import host-only code.
4. Tests that depend on tokio/mio networking or file descriptors.
5. `wasm-bindgen-test-runner` killed ChromeDriver before it bound a port because ChromeDriver logged a warning on stderr (`FromSockAddr failed on netmask`). ✅ default `CHROMEDRIVER_ARGS=--log-level=SEVERE` so the warning never emits and the runner stops recycling the driver.
6. New runtime blocker after the driver fix: plotting tests panic with `one-time initialization may not be performed recursively` because `plotting::tests::ensure_plot_test_env()` wraps another `Once`. Need to replace it with `OnceLock<()>` or refactor the helper to avoid nested initialization.

## Work Log

| Date | Change | Notes |
| ---- | ------ | ----- |
| 2025-12-12 | BLAS/LAPACK tests host-only | Entire file now `cfg(all(feature="blas-lapack", not(target_arch="wasm32")))` so wasm builds skip Accelerate/OpenBLAS while native coverage remains intact. |
| 2025-12-12 | Replaced plotting `#[ctor]` hooks | Added `plotting::tests::ensure_plot_test_env()` (Once + `disable_rendering_for_tests`) and invoked it at the start of each plotting test; removes `ctor` dependency and keeps both native + wasm test behavior identical. |
| 2025-12-12 | Stabilized ChromeDriver startup | `CHROMEDRIVER_ARGS` now defaults to `--log-level=SEVERE`, suppressing macOS `FromSockAddr` warnings that `wasm-bindgen-test-runner` misinterpreted as driver failure. Runtime wasm tests progress past driver startup; current failure is the plotting `Once` recursion panic. |

