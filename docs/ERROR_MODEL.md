# RunMat Error Model (Rust-native, MATLAB-compatible surface)

## Purpose
RunMat’s error system should deliver **Rust-compiler‑grade diagnostics** while preserving MATLAB
compatibility **only at the user‑facing boundary**. Internally, errors are typed, structured, and
propagated with standard Rust mechanisms. This eliminates string‑based sentinels and unlocks rich
context, tracing, and precise source attribution.

This document defines the **final model** and the principles that guide builtin and runtime
behavior.

## Design goals
- **Typed errors end‑to‑end**: no string‑only error plumbing in core runtime or builtins.
- **Diagnostic‑rich output**: use `miette`‑style diagnostics with spans, labels, notes, and help.
- **Async‑native execution**: async/await is the only suspension mechanism.
- **MATLAB compatibility at the surface only**: convert structured errors to `MException` where
  required by MATLAB semantics, not as an internal representation.
- **Context‑preserving propagation**: maintain builtin name, call stack, and span context.

## Core types and layering

### Builtin results
Builtins return:

```rust
pub type BuiltinResult<T> = Result<T, RuntimeError>;
```

`RuntimeError` is defined in **`runmat-async`** (to avoid dependency cycles) and re-exported by
`runmat-runtime`.

### Structured error payload
`RuntimeError` implements `std::error::Error`, `thiserror::Error`, and `miette::Diagnostic`, and exposes `format_diagnostic()` for Rust-style CLI/REPL rendering.

Recommended shape:
- **Message**: human‑readable primary text.
- **Span**: optional `SourceSpan` for location labels.
- **Source**: optional chained error.
- **Context**: builtin name, call stack, execution phase.
- **Identifier**: optional MATLAB‑style identifier for surface compatibility.

### Context capture (builder pattern)
Use a builder API (defined in `runmat-runtime`) to standardize context without churn at call sites.

Example:

```rust
let err = runtime_error("unsupported input type")
    .with_identifier("MATLAB:array:invalidType")
    .with_builtin("flip")
    .with_call_stack(call_stack)
    .with_span(span)
    .with_source(source);
```

## Boundary compatibility
MATLAB compatibility is required at the **language surface** only:
- `try/catch` and `MException` creation should remain consistent with MATLAB behavior.
- Internal errors are converted to `MException` only when surfacing to MATLAB semantics.
- Uncaught errors should render as Rust-style diagnostics (via `RuntimeError::format_diagnostic()`), with MATLAB identifiers surfaced as a separate labeled field.

## Propagation rules
**Do**
- Use `?` to propagate errors.
- Use `From`/`Into` to lift errors into the structured type.
- Preserve context (builtin, identifier, call stack) when mapping errors.

**Don’t**
- Convert structured errors to `String` in runtime/builtins.
- Encode control flow via sentinel strings.
- Hide spans or source chains when mapping errors.

## Summary
RunMat’s error model prioritizes **typed, structured diagnostics** internally and preserves MATLAB
semantics only at the final boundary. This unlocks **Rust‑style clarity and attribution** while
remaining fully compatible with existing MATLAB code at the user surface.
