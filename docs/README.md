# RunMat Runtime Documentation

RunMat is a high-performance runtime for MATLAB-syntax code.

This is the technical documentation for the runtime: the compiler pipeline, virtual machine, GPU and JIT tiers, session engine, plotting, and the host integration surfaces. For the MATLAB function reference and product pages, see [runmat.com](https://runmat.com).

## Start Here

New to RunMat? Follow these in order:

1. [Installation](./getting-started/install.md) — install the CLI or download Desktop.
2. [Hello World](./getting-started/hello-world.md) — write and run your first script.
3. [Command Line Interface](./getting-started/cli.md) — the REPL, running scripts, and runtime flags.
4. [Projects](./getting-started/projects.md) — multi-file projects, packages, classes, and entrypoints.
5. [Language Compatibility](./getting-started/compatability.md) — what MATLAB syntax and semantics are supported.

If you want to understand how the runtime works internally, start with the [Compilation Pipeline](./compiler/index.md) and follow the cross-links from there.

## Getting Started

- [Installation](./getting-started/install.md)
- [Desktop](./getting-started/desktop.md)
- [Command Line Interface](./getting-started/cli.md)
- [Projects](./getting-started/projects.md)
- [Configuration](./getting-started/config.md)
- [Language Compatibility](./getting-started/compatability.md)
- [Hello World](./getting-started/hello-world.md)

## Reference

- [Changelog](./CHANGELOG.md)
- [Values & Type Model](./VALUES.md)
- [Glossary](./GLOSSARY.md)

## Compilation Pipeline

- [Compilation Pipeline](./compiler/index.md)
- [Lexer & Parser](./compiler/lexer-and-parser.md)
- [High-Level IR (HIR)](./compiler/hir.md)
- [Module Composition](./compiler/modules.md)
- [Mid-Level IR (MIR)](./compiler/mir.md)
- [MIR & Static Analysis](./compiler/static-analysis.md)

## Virtual Machine (VM)

- [Virtual Machine (VM)](./vm/index.md)
- [Bytecode Compilation (MIR to Bytecode)](./vm/bytecode.md)
- [Interpreter Dispatch & Execution Loop](./vm/interpreter.md)
- [Indexing Subsystem](./vm/indexing.md)
- [Callable Resolution & Function Dispatch](./vm/dispatch.md)

## GPU Acceleration & Fusion Engine

- [GPU Acceleration & Fusion Engine](./gpu/index.md)
- [Fusion Engine & Residency Management](./gpu/fusion.md)
- [wgpu Backend & Accelerate Provider](./gpu/wgpu.md)

## JIT Compiler

- [JIT Compiler](./jit/index.md)
- [JIT Pipeline](./jit/pipeline.md)

## Builtins

- [Builtins](./builtins/index.md)
- [Authoring Builtins](./builtins/authoring.md)

## Session Engine

- [Session Engine](./session/index.md)
- [Execution Requests](./session/execution-requests.md)
- [Workspace State](./session/workspace.md)
- [Variable Inspection](./session/variable-inspection.md)
- [Snapshots & Replay](./session/snapshots.md)
- [Interaction & Streams](./session/interaction-and-streams.md)
- [Host Integration](./session/host-integration.md)

## Plotting System

- [Plotting System](./plotting/index.md)
- [Figure State & Handles](./plotting/state-and-handles.md)
- [Rendering Pipeline](./plotting/rendering.md)
- [Replay & Export](./plotting/replay-and-export.md)
- [Host Integration](./plotting/host-integration.md)

## WebAssembly & TypeScript

- [WASM & TypeScript/JavaScript](./wasm/index.md)

## Language Server Protocol (LSP)

- [Language Server Protocol (LSP)](./lsp/index.md)
- [Diagnostics & Highlighting](./lsp/diagnostics-and-highlighting.md)
- [Editor Features](./lsp/features.md)

## Memory Management

- [Memory Management](./gc/index.md)

## Execution

- [Execution](./execution/index.md)
- [Async Execution](./execution/async.md)
- [Errors & Diagnostics](./execution/errors.md)

## Filesystem

- [Filesystem Abstraction](./fs/index.md)
- [Datasets API](./fs/datasets.md)

## Development

- [Development](./development/index.md)
- [Build System](./development/build-system.md)
- [Supported Architectures](./development/supported-architectures.md)
- [Testing Strategy](./development/testing.md)
- [Benchmarking](./development/benchmarking.md)
- [Telemetry](./development/telemetry.md)