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

The detailed architecture lives in `docs/architecture.md`. Consult it for the
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
- [ ] Simple interpreter running on an unoptimised bytecode (`rustmat-ignition`).
- [ ] Headless plotting backend emitting SVG/PNG.
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

### 2025-07-28
- Repository initialised with lexer crate, repl placeholder and docs.

### 2025-07-29
- Expanded lexer with operators, comments and tests.

### 2025-07-30
- Added error token, case-sensitivity tests and failure handling in lexer.

### 2025-07-31
- Introduced `rustmat-parser` crate with simple AST and parser tests.

### 2025-08-01
- Added basic rustmat-repl that tokenizes input and prints tokens. Included integration tests.

### 2025-08-02
- Expanded REPL test suite with edge cases and error handling.
- Updated docs to describe the tokenizing behaviour.

### 2025-08-03
- Fixed clippy warning in `format_tokens` and added binary test.

### 2025-08-04
- Removed `assert_cmd` and `predicates` dev dependencies to avoid pulling in an
  extra version of `regex-syntax`. Rewrote the REPL binary test using
  `std::process::Command`.

### 2025-08-05
- Enabled GitHub Actions CI to run `cargo fmt`, `cargo check`, `clippy` and tests.
  The P0 milestone tasks are now complete.

### 2025-08-06
- Implemented precedence climbing parser with unary operators and power
  expression. Added matrix literals with comma and semicolon separators.
- Extended test suite with comprehensive cases (now 20+). This completes
  the remaining P1 parser work.

### 2025-08-07
- Addressed review feedback: added coverage for left division and elementwise
  operators bringing parser tests above twenty cases.
- Marked the parser milestone complete and noted future grammar tasks
  (control flow, functions, indexing).

### 2025-08-08
- Extended parser with control flow statements, function definitions and array
  indexing. Added tests exercising these features so typical MATLAB files parse
  without errors.

### 2025-08-09
- Introduced `rustmat-hir` crate implementing high-level IR with scope and type
  annotations. Includes translation from AST, simple type inference and error
  handling for undefined variables. Added comprehensive tests covering normal
  cases, failures and scope edge cases.

### 2025-08-10
- Extended HIR lowering with variable type tracking and updated tests to verify
  inference across assignments and redefinitions.
