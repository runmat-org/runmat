# RunMat Error Model (Rust-native, MATLAB-compatible surface)

## Purpose
RunMat’s error system should deliver **Rust-compiler‑grade diagnostics** while preserving MATLAB
compatibility **only at the user‑facing boundary**. Internally, errors must be typed, structured, and
propagated with standard Rust mechanisms. This eliminates string‑based sentinels and unlocks rich
context, tracing, and precise source attribution.

This document defines the target model and the migration principles that guide builtin and runtime
changes.

## Design goals
- **Typed errors end‑to‑end**: no string‑only error plumbing in core runtime or builtins.
- **Diagnostic‑rich output**: use `miette`‑style diagnostics with spans, labels, notes, and help.
- **Async‑safe control flow**: `Suspend` is a control‑flow signal and must never be stringified.
- **MATLAB compatibility at the surface only**: convert structured errors to `MException` where
  required by MATLAB semantics, not as an internal representation.
- **Context‑preserving propagation**: maintain builtin name, task id, and call stack context for
  traceability.

## Core types and layering
### Runtime control flow
Builtins continue to return:

```rust
pub type BuiltinResult<T> = Result<T, RuntimeControlFlow>;
```

`RuntimeControlFlow` carries suspension or errors:
- `Suspend(...)` is never converted to strings.
- `Error(...)` carries a **structured error type**, not a `String`.

### Structured error payload
`RuntimeError` is defined in **`runmat-async`** (to avoid dependency cycles) and re-exported by
`runmat-runtime`. It implements `std::error::Error`, `thiserror::Error`, and `miette::Diagnostic`.

Recommended shape:
- **Message**: human‑readable primary text.
- **Span**: optional `SourceSpan` for location labels.
- **Source**: optional chained error.
- **Context**: builtin name, task id, call stack, execution phase.
- **Identifier**: optional MATLAB‑style identifier for surface compatibility.

This structure enables:
- Precise diagnostics at runtime.
- Clean propagation via `?` and `From`.
- Rich logging and tracing integration.

### Context capture (builder pattern)
Use a builder API (defined in `runmat-runtime`) to standardize and extend context without churn at
call sites. Provide a lightweight helper that returns the builder for the common case.

Example:

```rust
let err = runtime_error("unsupported input type")
    .with_identifier("MATLAB:array:invalidType")
    .with_builtin("flip")
    .with_task_id(task_id)
    .with_call_stack(call_stack)
    .with_span(span)
    .with_source(source);
```

Notes:
- `task_id` should be attached when available; it can be `None` for sync paths.
- `call_stack` should be captured consistently at the runtime boundary.
- The builder should remain cheap to call and easy to extend.

## Boundary compatibility
MATLAB compatibility is required at the **language surface** only:
- `try/catch` and `MException` creation should remain consistent with MATLAB behavior.
- Internal errors are converted to `MException` only when surfacing to MATLAB semantics.
- Uncaught errors should render as `miette` diagnostics, with MATLAB identifiers preserved in the
  structured error for consistency.

This approach preserves legacy behavior while enabling much richer diagnostics for RunMat users.

## Diagnostics and tracing
The runtime should render errors using `miette` formatting, with:
- Primary label at the most relevant span (when available).
- Notes for builtin name, task id, and call stack.
- Help messages where useful.

Errors should carry structured context so that diagnostics can be **attributable and traceable** in
complex async execution, without relying on fragile string parsing.

## Propagation rules
**Do**
- Use `?` to propagate errors.
- Use `From`/`Into` to lift errors into the structured type.
- Preserve `Suspend` without conversion.

**Don’t**
- Convert `RuntimeControlFlow` to `String` (this loses `Suspend`).
- Return `Err("...")` in builtins; wrap into the structured error type instead.
- Encode control flow via sentinel strings.

## Migration status
`RuntimeError` and the builder API live in `runmat-runtime`, but the wider runtime still uses
string‑backed `RuntimeControlFlow` in many places. Migration will proceed folder‑by‑folder to switch
error plumbing to structured diagnostics.

## Migration guidance (builtins)
When migrating a builtin or folder of builtins:
1) Identify string‑based error and suspension paths in the call stack.
2) Replace `Result<T, String>` with `BuiltinResult<T>` or a structured error result.
3) Use `?` to propagate errors; avoid `map_err(|e| e.to_string())`.
4) Preserve error context (builtin name, identifier, call stack) in the structured error.
5) Ensure no `RuntimeControlFlow` to string conversions remain.

## Open questions / follow‑ups
- How to attach source spans in the interpreter (PC → source span mapping)?
- When to flip `RuntimeControlFlow::Error` to structured errors across the runtime.

## Summary
RunMat’s error model prioritizes **typed, structured diagnostics** internally and preserves MATLAB
semantics only at the final boundary. This unlocks **Rust‑style clarity and attribution** while
remaining fully compatible with existing MATLAB code at the user surface.
