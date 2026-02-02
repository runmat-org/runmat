# Async Runtime Migration Status

This document captures the **final state** of the async runtime migration. The transitional
control‑flow layer is gone, and execution is fully async with structured Rust errors.

## Final model summary

- **Async execution end‑to‑end**: `RunMatSession::execute` and the ignition interpreter are `async`.
- **Builtins return `RuntimeError`**: the runtime API is now `Result<T, RuntimeError>` for all
  builtins and helpers.
- **No control‑flow enum**: `RuntimeControlFlow`, `Suspend`, and resume loops are removed.
- **Input is handler‑driven**: input handlers return `Result<InputResponse, String>` and are
  converted to `RuntimeError` internally.
- **String conversions are edge‑only**: only the CLI/UI layers format errors as strings.

## Completed migration steps

- Converted interpreter entry points (`interpret`/`execute`) to async.
- Switched builtin dispatch to async call sites.
- Removed suspend/resume state and pending frames across ignition/core.
- Migrated builtins to return `RuntimeError` directly.
- Updated wasm/cli/kernel callers to await async execution.

## Remaining follow‑ups (future work)

- **Awaitable GPU readback**: replace the current “pending map read” error with a future‑based
  awaitable in the GPU provider.
- **Language async/await**: add syntax and runtime support once the awaitable substrate is ready.
- **Cleanup unused scaffolding**: remove any leftover suspend markers if they stay unused.

## References

- `docs/ERROR_MODEL.md` for the structured error design.
- `FINAL_ASYNC_ERROR_MODEL.md` for the final async/error architecture summary and diagrams.
