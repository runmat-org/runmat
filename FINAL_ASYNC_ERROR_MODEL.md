# Final Async + Error Model

This document describes the **final runtime model** for RunMat after the async migration and error
system cleanup. It is the reference for how execution, errors, and I/O behave today.

## Overview

- **Async execution**: `RunMatSession::execute` is `async` and drives the ignition VM.
- **Typed errors**: all runtime and builtin errors are `RuntimeError`.
- **No control‑flow enums**: suspension is handled by Rust `await`, not sentinel values.
- **MATLAB compatibility**: structured errors are converted to `MException` only at the language
  surface (try/catch and error display).

## Execution flow

```
Caller (CLI/WASM/Kernel)
        |
        v
RunMatSession::execute (async)
        |
        v
Ignition VM (async interpreter loop)
        |
        v
Builtin call (Result<Value, RuntimeError>)
```

Key points:
- The VM awaits builtins normally; there is no suspend/resume state.
- Execution results are packaged into `ExecutionResult` for the host.

## Error construction and propagation

```
Builtin/Helper Error
        |
        v
RuntimeError (builder adds context)
        |
        v
Ignition VM (propagates Result)
        |
        v
MATLAB surface (try/catch -> MException)
        |
        v
Host (CLI/WASM) formats diagnostics via `RuntimeError::format_diagnostic()`
```

Rules:
- Errors propagate with `?`.
- Only the surface formats errors into strings.
- VM maps errors into `MException` when executing `try/catch` or `error()`.

## RuntimeError structure

```
RuntimeError
  ├─ message: String
  ├─ identifier: Option<String>
  ├─ span: Option<SourceSpan>
  ├─ source: Option<Box<dyn Error>>
  └─ context: ErrorContext { builtin, call_stack, phase, ... }
```

Context is attached via the builder API (`build_runtime_error`).

## Input handling

```
Builtin -> interaction::request_line / wait_for_key
        |
        v
Input handler (sync or async hook)
        |
        v
Result<InputResponse, String>
        |
        v
RuntimeError (if error) or InputResponse
```

- Handlers are installed by the host (`RunMatSession::install_input_handler`).
- Handler errors are converted to `RuntimeError` within the runtime boundary.
- There is no “pending” control‑flow; handlers must respond or error.

## GPU readback behavior (current)

- GPU map/readback that is still pending returns a **runtime error**.
- Follow‑up work will replace this with an awaitable GPU future (see `docs/ARCH_ASYNC_PLAN.md`).

## Boundary formatting

```
RuntimeError
   |
   v
CLI / WASM / Kernel
   |
   v
Formatted message or diagnostic output (`RuntimeError::format_diagnostic()`)
```

The boundary is the only place where errors are stringified.

## Summary

RunMat now runs on **async execution** with **structured errors** everywhere. There are no
string‑sentinel or control‑flow enums in the runtime path. Errors are precise, contextual, and
converted to MATLAB‑compatible exceptions only when required by the language surface.
