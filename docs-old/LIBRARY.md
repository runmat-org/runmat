# RunMat Standard Library Design

This document describes how built-in MATLAB functions are implemented in the
RunMat project. The goal is to mirror the behaviour of the MATLAB language
while keeping the codebase maintainable and approachable.

## Organization

Each group of built-ins lives in its own crate under `crates/`. For example
`runmat-plot` implements plotting related functions. Future crates will cover
math operations, file I/O and more. Functions are written in plain Rust and
exposed to the runtime via a thin attribute macro. Metadata about each builtin
is stored in the `runmat-builtins` crate using the `inventory` pattern so the
runtime can populate its global context automatically.

## Declaring Functions

Use the `matlab_fn` attribute from the `runmat-macros` crate to clearly mark the
MATLAB name for every exported function. The macro registers the function in the
`runmat-builtins` inventory so the runtime can discover and document it:

```rust
use runmat_macros::matlab_fn;

#[matlab_fn(name = "plot")]
pub fn plot_line(xs: &[f64], ys: &[f64], path: &str) -> Result<(), String> {
    // implementation
}
```

The attribute attaches documentation so readers immediately know which MATLAB
builtin is implemented and records the mapping for the runtime. Additional
metadata can be added later without rewriting existing code.

## Adding New Crates

1. Create a new crate under `crates/` using kebab-case naming (e.g.
   `runmat-linear-algebra`).
2. Add the crate to the workspace members in the root `Cargo.toml`.
3. Implement the functions, annotating each with `#[matlab_fn(name = "...")]`.
4. Provide comprehensive unit tests covering typical usage, error cases and edge
   conditions. Tests live in `crates/<name>/tests/`.
5. Update `PLAN.md` with a short entry summarising the addition.

## Testing Guidelines

Tests should exercise both success and failure paths. Prefer temporary files and
in-memory data so tests remain hermetic. Running `cargo test --all` from the
repository root must succeed without network access.
For macros or other compile-time utilities, use the `trybuild` crate to verify
that misuse results in helpful compile errors.

## Style

- Follow Rustfmt defaults (`cargo fmt`).
- Keep functions small and focused.
- Document any deviations from MATLAB behaviour in code comments.

Following this methodology ensures a consistent, readable and easily extensible
standard library.
