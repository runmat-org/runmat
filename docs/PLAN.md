# Development Plan

This document serves as the evolving blueprint for **RustMat**. It should retain
all notes from prior work so future contributors can trace our rationale. Append
new dated sections at the end rather than rewriting history.

## Methodology

- **Read this plan from top to bottom before starting work.** It records design
decisions, open tasks and progress logs.
- **Write a short log entry** for each pull request summarizing what changed and
any observations or blockers.
- **Keep tasks organised by milestone.** Mark tasks as `[x]` once complete.
- **Preserve context.** When refactoring or changing direction, add a new entry
explaining why.
- **Run `cargo check`** before committing and document the result in the PR.

The detailed architecture lives in `docs/ARCHITECTURE.md`. Consult it for the
rationale behind each crate and subsystem.

## Roadmap

RustMat adopts a V8-inspired tiered architecture: a baseline interpreter called
*Ignition* feeds profiling data to an optimising JIT (*Turbine*). A snapshot of
the standard library enables fast start-up. The workspace is split into many
kebab-case crates (lexer, parser, IR passes, runtime, GC, JIT, kernel, etc.).

### Milestone P0 – Bootstrapping

- [x] Expand lexer to cover MATLAB operators, keywords and comments.
- [x] Introduce `rustmat-parser` crate producing an AST.
- [x] Basic `rustmat-repl` that tokenizes input and prints tokens.
- [x] Set up CI with `cargo fmt` and `cargo check`.

### Milestone P1 – Language Core

- [x] Complete parser with precedence rules and matrix syntax.
 - [x] Extend parser to support control flow, function definitions and array
      indexing so that typical MATLAB files can be parsed without errors.
- [x] High-level IR (`rustmat-hir`) with scope and type annotations.
- [x] Simple interpreter running on an unoptimised bytecode (`rustmat-ignition`).
- [x] Headless plotting backend emitting SVG/PNG.
- [ ] Jupyter kernel communication skeleton.

### Milestone P2 – Performance Features

- [ ] Cranelift-based JIT (`rustmat-turbine`).
- [ ] BLAS/LAPACK bindings and array primitives (`rustmat-runtime`).
- [ ] Generational GC with optional pointer compression (`rustmat-gc`).
- [ ] Snapshot creator to preload the standard library (`rustmat-snapshot`).

### Milestone P3 – User Experience

- [ ] GUI plotting via WGPU.
- [ ] MEX/C API (`rustmat-ffi`).
- [ ] Package manager stub and documentation website.

## Log

### Edit 1
- Repository initialised with lexer crate, repl placeholder and docs.

### Edit 2
- Expanded lexer with operators, comments and tests.

### Edit 3
- Added error token, case-sensitivity tests and failure handling in lexer.

### Edit 4
- Introduced `rustmat-parser` crate with simple AST and parser tests.

### Edit 5
- Added basic rustmat-repl that tokenizes input and prints tokens. Included integration tests.

### Edit 6
- Expanded REPL test suite with edge cases and error handling.
- Updated docs to describe the tokenizing behaviour.

### Edit 7
- Fixed clippy warning in `format_tokens` and added binary test.

### Edit 8
- Removed `assert_cmd` and `predicates` dev dependencies to avoid pulling in an
  extra version of `regex-syntax`. Rewrote the REPL binary test using
  `std::process::Command`.

### Edit 9
- Enabled GitHub Actions CI to run `cargo fmt`, `cargo check`, `clippy` and tests.
  The P0 milestone tasks are now complete.

### Edit 10
- Implemented precedence climbing parser with unary operators and power
  expression. Added matrix literals with comma and semicolon separators.
- Extended test suite with comprehensive cases (now 20+). This completes
  the remaining P1 parser work.

### Edit 11
- Addressed review feedback: added coverage for left division and elementwise
  operators bringing parser tests above twenty cases.
- Marked the parser milestone complete and noted future grammar tasks
  (control flow, functions, indexing).

### Edit 12
- Extended parser with control flow statements, function definitions and array
  indexing. Added tests exercising these features so typical MATLAB files parse
  without errors.

### Edit 13
- Introduced `rustmat-hir` crate implementing high-level IR with scope and type
  annotations. Includes translation from AST, simple type inference and error
  handling for undefined variables. Added comprehensive tests covering normal
  cases, failures and scope edge cases.

### Edit 14
- Extended HIR lowering with variable type tracking and updated tests to verify
  inference across assignments and redefinitions.

### Edit 15
- Fixed clippy warnings in `rustmat-hir` after review.

### Edit 16
- Implemented `rustmat-ignition` crate with a simple bytecode interpreter. The
  interpreter supports numeric operations, variable assignments, `if`, `while`
  and `for` loops with break/continue. Added comprehensive tests covering
  normal execution, error cases and edge conditions. All workspace tests pass.

### Edit 17
- Added support for `elseif` branches in the Ignition compiler and updated the
  interpreter tests accordingly. Marked the interpreter milestone complete.

### Edit 18
- Expanded interpreter tests with break/continue error cases, nested loops, multiple elseif branches and return behaviour.
- All workspace tests pass.

### Edit 19
- Addressed CI failure by removing a boolean comparison in the interpreter tests.
- Verified `cargo check`, `cargo clippy` and `cargo test` all succeed.

### Edit 20
- Implemented `rustmat-plot` crate providing headless SVG/PNG rendering with
  configurable defaults loaded via `RUSTMAT_PLOT_CONFIG`.
- Added integration tests covering config loading, rendering and invalid style
  handling. All workspace tests pass.

### Edit 21
- Expanded `rustmat-plot` with scatter, bar and histogram plots and per-type
  styling options loaded from YAML. Updated integration tests to cover the new
  functions and fixed a bar chart label bug. All checks pass.

### Edit 22
- Added stub 3D plotting APIs (`plot_3d_scatter` and `plot_surface`) returning
  errors as Plotters lacks native 3D support.
- Extended the `rustmat-plot` test suite with numerous positive and negative
  cases, including validation of error paths. All workspace tests pass.

### Edit 23
- Replaced the 3D stubs with working implementations using a simple isometric
  projection so scatter and surface plots now render to SVG/PNG.
- Added integration tests covering the new 3D functions and error cases for
  length mismatch and non-square grids.

### Edit 24
- Introduced `rustmat-macros` crate with a `matlab_fn` attribute to mark
  standard library functions.
- Annotated all plotting functions with their MATLAB names and documented the
  library design in `docs/LIBRARY.md`.

### Edit 25
- Extended the `matlab_fn` macro to register builtins via the new
  `rustmat-builtins` crate using `inventory`.
- Added tests verifying registration and updated docs to explain the process.

### Edit 26
- Added comprehensive tests for the builtin registry and `matlab_fn` macro.
  Compile-fail checks ensure attribute misuse is caught at build time.
  New tests validate documentation capture and registration across modules.
- Addressed clippy errors in `rustmat-macros` and `rustmat-plot` after review.

### Edit 27
- Replaced the isometric 3D projection with a perspective transform and updated
  the plotting code accordingly.
- Updated integration tests and documentation to reflect the new approach.

### Edit 28
- Removed clippy allowances in the plotting crate by computing plot bounds
  internally.
- Verified that perspective-based 3D plotting works without lint overrides and
  all tests pass.

### Edit 29
- Disabled Plotters default features so tests no longer require the system `fontconfig`
  library. PNG output now relies on the `bitmap_encoder` feature only.
- Confirmed `cargo clippy` and the full test suite pass in a minimal environment.
