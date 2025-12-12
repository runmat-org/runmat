# Browser Test Harness

We can now exercise the crates that ship inside the `runmat-wasm` bundle directly in
headless Chrome via `wasm-pack test`. The entry point is `scripts/test-wasm-headless.sh`.

```bash
cd runmat
scripts/test-wasm-headless.sh
```

What the script does:

1. Regenerates the wasm builtin registry (`RUNMAT_GENERATE_WASM_REGISTRY=1 cargo check …`).
2. Runs `wasm-pack test --chrome --headless` for each crate we currently support
   (`runmat-core` by default).
3. Uses `scripts/chrome-headless.sh` to start Chrome/Chromium with the right WebGPU flags
   (`--headless=new --enable-unsafe-webgpu --use-angle=metal|vulkan`).

## Configuration

Environment variables:

| Variable | Description |
| --- | --- |
| `RUNMAT_CHROME_BIN` | Override the Chrome binary (defaults to the system Chrome/Chromium). |
| `RUNMAT_WASM_INCLUDE_RUNTIME=1` | Attempt to run the `runmat-runtime` tests as well (see below). |
| `WASM_BINDGEN_TEST_TIMEOUT` | Per-test timeout passed to `wasm-bindgen-test` (default `120`). |

## Current coverage

- `runmat-core` — all unit and integration tests now compile for `wasm32-unknown-unknown`
  and execute headlessly in Chrome.

- `runmat-runtime` — **work in progress.** Enabling this crate today requires gating:
  - BLAS/LAPACK tests that rely on native Accelerate/OpenBLAS (`tests/blas_lapack.rs`).
  - Plotting ops that use the `ctor` macro (unsupported on wasm).
  - Doc-block tests that import `test_support` unconditionally.
  - Async/runtime tests that depend on `tokio`, `mio`, or host networking.

Set `RUNMAT_WASM_INCLUDE_RUNTIME=1` to see the current failure list; we will tighten the
feature flags and/or guard those tests in follow-up work.

## Future steps

- Extend the script to cover additional crates (`runmat-snapshot`, `runmat-plot`, etc.) once
  their tests are wasm-safe.
- Add a CI job that runs `scripts/test-wasm-headless.sh` against Chrome-for-Testing on Linux.
- Continue migrating existing tests by adding `#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]`
  (done via the regex-based helper in this change) or gating them when they rely on native-only
  functionality.

