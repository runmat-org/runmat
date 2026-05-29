# RunMat Runtime Documentation

- [Changelog](./CHANGELOG.md)
- [Values & Type Model](./VALUES.md)
- [Glossary](./GLOSSARY.md)

## Getting Started

- [Installation](./getting-started/install.md)
- [Command Line Interface](./getting-started/cli.md)
- [Configuration](./getting-started/config)
- [Language Compatability](./getting-started/compatability.md)
- [Hello World](./getting-started/hello-world.md)

## Compilation Pipeline

- [Compilation Pipeline](./compiler)
- [Lexer & Parser](./compiler/lexer-and-parser.md)
- [High-Level IR (HIR)](./compiler/hir.md)
- [Mid-Level IR (MIR)](./compiler/mir.md)
- [MIR & Static Analysis](./compiler/static-analysis.md)

## Virtual Machine (VM)

- [Virtual Machine (VM)](./vm)
- [Bytecode Compilation (MIR → Bytecode)](./vm/bytecode)
- [Interpreter Dispatch & Execution Loop](./vm/interpreter.md)
- [Indexing Subsystem](./vm/indexing.md)
- [Callable Resolution & Function Dispatch](./vm/dispatch.md)

## GPU Acceleration & Fusion Engine

- [GPU Acceleration & Fusion Engine](./gpu)
- [Fusion Engine & Residency Management](./gpu/fusion.md)
- [wgpu Backend & Accelerate Provider](./gpu/wgpu.md)

## JIT Compiler

- [JIT Compiler](./jit)
- [JIT Pipeline](./jit/pipeline.md)

## Builtins

- [Builtins](./builtins)
- [Authoring Builtins](./builtins/authoring.md)

## Session Engine

- [Session Engine](./session)
- [Execution Requests](./session/execution-requests.md)
- [Workspace State](./session/workspace.md)
- [Variable Inspection](./session/variable-inspection.md)
- [Snapshots & Replay](./session/snapshots.md)
- [Interaction & Streams](./session/interaction-and-streams.md)
- [Host Integration](./session/host-integration.md)

## Plotting System

- [Plotting System](./plotting)
- [Figure State & Handles](./plotting/state-and-handles.md)
- [Rendering Pipeline](./plotting/rendering.md)
- [Replay & Export](./plotting/replay-and-export.md)
- [Host Integration](./plotting/host-integration.md)

## WebAssembly & TypeScript

- [WASM & TypeScript/JavaScript](./wasm)

## Language Server Protocol (LSP)

- [Language Server Protocol (LSP)](./lsp)
- [Diagnostics & Highlighting](./lsp/diagnostics-and-highlighting.md)
- [Editor Features](./lsp/features.md)

## Memory Management

- [Memory Management](./gc)

## Execution

- [Execution](./execution)
- [Async Execution](./execution/async.md)
- [Errors & Diagnostics](./execution/errors.md)

## Filesystem

- [Filesystem Abstraction](./fs)
- [Datasets API](./fs/datasets.md)

## Development

- [Development](./development)
- [Build System](./development/build-system.md)
- [Supported Architectures](./development/supported-architectures.md)
- [Testing Strategy](./development/testing.md)
- [Benchmarking](./development/benchmarking.md)
- [Telemetry](./development/telemetry.md)
