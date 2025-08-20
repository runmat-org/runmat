# Development Roadmap

This document serves as the evolving roadmap for **RunMat** to track the progress of the project towards the 1.0 release. 

The core language and runtime are complete. The major remaining items are:

- The GUI plotting backend has a few edges that need to be smoothed out.
- The package manager needs to be implemented.
- The JIT currently optimizes a subset of the language, with an interpreter fallback for operations that are currently not implemented in the JIT. Full JIT coverage will be implemented before the 1.0 release.
- Some of the built-in functions have partial implementations. We will take a full pass through the built-in functions to ensure they're a good minimal set, with full canonical behavior.

The following is a list of milestones that we will be working on in the future. The milestones are not necessarily in order, and we will be working on multiple milestones at the same time.

The below tracks the progress of the project towards the 1.0 release to date.

### Milestone P0 - Bootstrapping

- [x] Expand lexer to cover MATLAB operators, keywords and comments.
- [x] Introduce `runmat-parser` crate producing an AST.
- [x] Basic `runmat-repl` that tokenizes input and prints tokens.
- [x] Set up CI with `cargo fmt` and `cargo check`.

### Milestone P1 - Language Core

- [x] Complete parser with precedence rules and matrix syntax.
 - [x] Extend parser to support control flow, function definitions and array
      indexing so that typical MATLAB files can be parsed without errors.
- [x] High-level IR (`runmat-hir`) with scope and type annotations.
- [x] Simple interpreter running on an unoptimised bytecode (`runmat-ignition`).
- [x] Headless plotting backend emitting SVG/PNG.
- [x] Jupyter kernel communication skeleton.

### Milestone P2 - Performance Features

- [x] Cranelift-based JIT (`runmat-turbine`).
- [x] BLAS/LAPACK bindings and array primitives (`runmat-runtime`).
- [x] Generational GC with optional pointer compression (`runmat-gc`).
- [x] Snapshot creator to preload the standard library (`runmat-snapshot`).
- [x] v0.1 release.

### Milestone P3 - User Experience

- [x] Full language coverage (`docs/LANGUAGE_COVERAGE.md`).
- [x] v0.2 release.
- [ ] Full standard library pass for canonical built-ins.
- [ ] Package manager.
- [ ] GUI plotting via WGPU.
- [ ] LSP server.
- [ ] Complete Accelerate (GPU) support.
- [ ] MEX/C API (`runmat-ffi`).