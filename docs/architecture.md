# Architecture Overview

RustMat aims to provide a high performance, MATLAB‑compatible runtime implemented in Rust. The project borrows extensively from the V8 JavaScript engine: a small baseline interpreter feeds type feedback to an optimising JIT, and the standard library can be snapshotted to minimise start‑up time.

## 1. Major Components

- **Front‑end** – implemented by `rustmat-lexer` and `rustmat-parser` for the MATLAB/Octave grammar.
- **IR** – represented by `rustmat-hir` and `rustmat-mir` for analysis and optimisation.
- **rustmat-ignition** – baseline byte‑code interpreter with inline caches.
- **rustmat-turbine** – optimising JIT using Cranelift or LLVM.
- **rustmat-runtime** – arrays, strings, file I/O and BLAS bindings.
- **rustmat-gc** – generational garbage collector with optional pointer compression.
- **rustmat-kernel** – Jupyter/IPython kernel running over ZMQ.
- **rustmat-repl** – stand‑alone command line interface.
- **rustmat-plot** – handle‑graphics style scene graph with pluggable renderers.

Additional crates such as `rustmat-pal` (platform abstraction) and `rustmat-ffi` (C/MEX bridge) mirror V8's PAL and embedder API.

## 2. V8‑Inspired Ideas

- **Ignition → Turbine Tiering** – code starts in a portable interpreter and hot functions are recompiled by the JIT for speed.
- **Hidden Classes & Inline Caches** – struct and object layouts are tracked to make property access constant time.
- **Feedback Vector** – byte‑code records type information that guides the optimiser.
- **Speculative Optimisation & Deopt** – the JIT inserts guards and falls back to the interpreter if assumptions fail.
- **Snapshot for Start‑up** – the build system can precompile the standard library into a binary blob.
- **Pointer Compression & Generational GC** – optional 32‑bit tagged pointers and a young/old heap keep memory usage low.
- **Isolates & Handles** – each kernel or REPL instance has an isolated heap managed through explicit handle scopes.
- **Platform Abstraction Layer** – wraps OS and CPU features so the runtime remains portable across x86‑64 and ARM64.

## 3. Runtime Stack

```
Jupyter / Editor / CLI
          │          (embedder API)
          ▼
rustmat-kernel (async ZMQ)
    │      └─ plot stream (SVG/PNG)
    ▼
rustmat-ignition ←deopt→ rustmat-turbine
    │  ▲
    │  └─ feedback / inline caches
    ▼
rustmat-runtime ⇐⇒ rustmat-pal + BLAS
          │
       rustmat-plot
```

Cold code runs in `rustmat-ignition`; after enough type feedback it is promoted to `rustmat-turbine`. Plots are rendered headless or via a GUI back-end depending on build features.

## 4. Workspace Layout (kebab‑case)

```
crates/
  rustmat-lexer/       # logos-based tokenizer
  rustmat-parser/      # LR parser producing the AST
  rustmat-ast/         # typed AST node definitions
  rustmat-hir/         # high-level IR with scopes and types
  rustmat-mir/         # SSA-style mid-level IR
  rustmat-bytecode/    # baseline bytecode format
  rustmat-ignition/    # interpreter executing the bytecode
  rustmat-ic/          # inline caches and feedback vectors
  rustmat-shapes/      # hidden class tracker for structs/arrays
  rustmat-turbine/     # optimising JIT via Cranelift/LLVM
  rustmat-gc/          # generational garbage collector
  rustmat-runtime/     # array types, BLAS, file I/O
  rustmat-pal/         # platform abstraction (threads, mmap)
  rustmat-snapshot/    # build-time snapshot generator
  rustmat-plot/        # handle-graphics scene and renderers
  rustmat-kernel/      # Jupyter kernel over ZMQ
  rustmat-repl/        # command-line REPL binary
  rustmat-ffi/         # C/MEX embedding interface
  rustmat-utils/       # shared logging and error helpers
```

Tests and examples live under `tests/` and `examples/` respectively. Feature flags such as `graphics-gui`, `jit-llvm` and `pointer-compression` keep the core slim.

## 5. Boot Sequence

1. **Build** – `cargo build --features snapshot` invokes `rustmat-snapshot` which parses the bundled `.m` files and serialises a heap image.
2. **Launch** – `rustmat-repl` or `rustmat-kernel` maps the snapshot, fixes pointers and enters the REPL in roughly 30 ms.
3. **Execution** – user code runs in `rustmat-ignition`; hot paths are JIT‑compiled by `rustmat-turbine`.

This document captures the overall design so new contributors can understand how the pieces fit together. More details about coding conventions and development workflows are found in `docs/DEVELOPING.md`.
