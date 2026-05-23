# Browser Test Harness

We can now exercise the crates that ship inside the `runmat-wasm` bundle directly in
headless Chrome via `wasm-pack test`. The entry point is `scripts/test-wasm-headless.sh`.

```bash
cd runmat
scripts/test-wasm-headless.sh
```

Replay-specific browser smoke tests can be run with:

```bash
cd runmat
scripts/test-wasm-replay-smoke.sh
```

What the script does:

1. Regenerates the wasm builtin registry (`RUNMAT_GENERATE_WASM_REGISTRY=1 cargo check …`).
2. Runs `wasm-pack test --chrome --headless` for each crate we currently support
   (`runmat-core` by default).
3. Uses `scripts/chrome-headless.sh` to start Chrome/Chromium with the right WebGPU flags
   (`--headless=new --enable-unsafe-webgpu --use-angle=metal|vulkan`).
4. Uses `scripts/resolve-chromedriver.sh` to pick a ChromeDriver compatible with the local
   Chrome version (or downloads one when allowed).

## Configuration

Environment variables:

| Variable | Description |
| --- | --- |
| `RUNMAT_CHROME_BIN` | Override the Chrome binary (defaults to the system Chrome/Chromium). |
| `CHROMEDRIVER_ARGS` | Extra args for the WebDriver binary (default `--log-level=SEVERE` to suppress early stderr warnings that confuse `wasm-bindgen-test`). |
| `RUNMAT_CHROMEDRIVER_BIN` | Explicit ChromeDriver binary to use (bypasses auto-resolution). |
| `RUNMAT_CHROMEDRIVER_ALLOW_DOWNLOAD` | Set to `0` to disable auto-download when no compatible cached driver exists (default `1`). |
| `RUNMAT_WASM_INCLUDE_RUNTIME=1` | Attempt to run the `runmat-runtime` tests as well (see below). |
| `WASM_BINDGEN_TEST_TIMEOUT` | Per-test timeout passed to `wasm-bindgen-test` (default `300`). |

If no compatible driver is found and download is disabled or unavailable, the scripts fall back
to wasm-pack's default ChromeDriver resolution.

For ad-hoc browser tests outside `scripts/test-wasm-headless.sh`:

```bash
cd runmat/crates/runmat-wasm
CHROMEDRIVER="$(../../scripts/resolve-chromedriver.sh)" \
  wasm-pack test --chrome --headless --chromedriver "${CHROMEDRIVER}" --test replay_smoke
```

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
- Keep the CI wasm browser job aligned with this script and Chrome-for-Testing on Linux.
- Continue migrating existing tests by adding `#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]`
  (done via the regex-based helper in this change) or gating them when they rely on native-only
  functionality.
