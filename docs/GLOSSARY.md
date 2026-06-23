---
title: "Glossary"
category: "Reference"
section: "glossary"
last_updated: "May 28, 2026"
---

# Glossary

This page defines RunMat-specific terms and abbreviations used across the runtime documentation. It focuses on terms that carry a specific meaning in this codebase.

## A

| Term | Definition | More |
| --- | --- | --- |
| ABI | The structured boundary between RunMat layers or between RunMat and a host. Session requests, workspace deltas, WASM payloads, and builtin descriptors are ABI surfaces because external callers depend on their shape. | [Session Engine](/docs/runtime/session) |
| Accelerate | RunMat's acceleration layer for GPU execution, provider hooks, and fusion-aware operations. In Cargo features and docs, `wgpu` is the currently wired backend while other backend feature names describe future or platform-specific integration points. | [GPU Acceleration & Fusion Engine](/docs/runtime/gpu) |
| Analysis store | The MIR analysis result container. It holds facts such as definite assignment, type and shape information, spawn safety, and async behavior that later compiler or runtime layers can query. | [MIR & Static Analysis](/docs/runtime/compiler/static-analysis) |
| AST | Abstract syntax tree. The parser produces the source-shaped representation before HIR lowering resolves scopes and bindings. | [Lexer & Parser](/docs/runtime/compiler/lexer-and-parser) |
| Async execution | RunMat's execution model for operations that may wait on host interaction, runtime futures, providers, or external I/O while preserving the session request boundary. | [Async Execution](/docs/runtime/execution/async) |

## B

| Term | Definition | More |
| --- | --- | --- |
| Backend | A concrete implementation behind a higher-level runtime surface, such as WGPU for GPU execution, BLAS/LAPACK for linear algebra, or a filesystem provider for storage. | [Build System](/docs/runtime/development/build-system) |
| Basic block | A linear MIR block ending in a terminator. MIR uses basic blocks to make control flow explicit for analysis and bytecode compilation. | [Mid-Level IR (MIR)](/docs/runtime/compiler/mir) |
| Binding | A resolved source name. HIR assigns binding IDs, MIR and bytecode carry those identities forward, and the session maps interactive workspace bindings to stable host-visible keys. | [High-Level IR (HIR)](/docs/runtime/compiler/hir) |
| BLAS/LAPACK | Native numerical libraries used by selected builtin linear algebra paths. They are build-time dependencies when the relevant Cargo features are enabled. | [Builtins](/docs/runtime/builtins) |
| Builtin | A MATLAB-visible function implemented by RunMat's Rust runtime. Builtins are registered with metadata so the VM, JIT, LSP, docs, and validation paths can reason about them consistently. | [Builtins](/docs/runtime/builtins) |
| Builtin descriptor | Structured metadata for a builtin: signatures, output behavior, completion policy, known errors, documentation text, and acceleration tags. | [Authoring Builtins](/docs/runtime/builtins/authoring) |
| Bytecode | The compact instruction form executed by the VM interpreter and used as the input for eligible JIT compilation. It is emitted from analyzed MIR. | [Bytecode Compilation](/docs/runtime/vm/bytecode) |

## C

| Term | Definition | More |
| --- | --- | --- |
| Callable descriptor | The runtime value used by the VM to invoke a resolved call target. It can represent builtins, bytecode functions, closures, handles, object methods, or fallback names. | [Callable Resolution & Function Dispatch](/docs/runtime/vm/dispatch) |
| Callable identity | The compiler-side description of what a call refers to. It records whether the target is already known, name-shaped, anonymous, local, external, or method-like. | [Callable Resolution & Function Dispatch](/docs/runtime/vm/dispatch) |
| Cell array | MATLAB container whose elements can hold independent values. RunMat cells own their element `Value`s directly; individual elements may still contain GC handles when identity or sharing must be preserved. | [Memory Management](/docs/runtime/gc) |
| CLI | The native `runmat` command-line interface. It hosts script execution, the REPL, configuration loading, telemetry setup, plotting integration, and developer commands. | [Installation](/docs/runtime/getting-started/install) |
| Compatibility mode | A runtime or request setting that selects MATLAB compatibility behavior. The session passes it into compilation and execution so diagnostics and lowering decisions can match the requested mode. | [Configuration](/docs/runtime/getting-started/config) |
| Completion | An LSP feature that offers names such as builtins, local variables, project functions, keywords, and properties from the current document context. | [Editor Features](/docs/runtime/lsp/features) |
| Control-flow graph | The graph formed by MIR basic blocks and terminators. It is the representation used for dataflow analysis and structured lowering into bytecode. | [Mid-Level IR (MIR)](/docs/runtime/compiler/mir) |

## D

| Term | Definition | More |
| --- | --- | --- |
| Dataflow analysis | MIR analysis that propagates facts across the control-flow graph until the facts stabilize. RunMat uses dataflow for assignment, type/shape, spawn-safety, and async behavior facts. | [MIR & Static Analysis](/docs/runtime/compiler/static-analysis) |
| Dataset | A named, chunked array stored through the filesystem layer so large data can be reopened, sliced, and updated without rewriting a whole file. | [Datasets API](/docs/runtime/fs/datasets) |
| Diagnostic | A structured parser, semantic, compile, runtime, or LSP message with a stable code and optional source span. Diagnostics are the basis for editor underlines and host-facing error payloads. | [Errors & Diagnostics](/docs/runtime/execution/errors) |
| Dispatch | The runtime process of selecting the correct operation for a bytecode instruction, call target, builtin, object method, or indexing form. | [Interpreter Dispatch & Execution Loop](/docs/runtime/vm/interpreter) |
| Document symbol | An LSP outline entry for functions, classes, sections, or other document structure that an editor can show in a symbols pane. | [Editor Features](/docs/runtime/lsp/features) |

## E

| Term | Definition | More |
| --- | --- | --- |
| End expression | MATLAB indexing syntax that refers to the size of the indexed dimension. RunMat keeps `end`-aware slice expressions explicit through MIR and bytecode so runtime shape can resolve them correctly. | [Indexing Subsystem](/docs/runtime/vm/indexing) |
| Execution outcome | The host-facing result of a session request. It contains values, workspace deltas, diagnostics, streams, figures, profiling data, effects, and optional fusion metadata. | [Execution Requests](/docs/runtime/session/execution-requests) |
| Execution request | The structured input submitted to `RunMatSession::execute_request`. It carries source, source identity, compatibility mode, host policy, workspace handle, and output preferences. | [Execution Requests](/docs/runtime/session/execution-requests) |

## F

| Term | Definition | More |
| --- | --- | --- |
| Feature flag | A Cargo or package feature that controls optional runtime surfaces such as `gui`, `wgpu`, `jit`, `blas-lapack`, WASM GPU support, or LSP WASM builds. | [Build System](/docs/runtime/development/build-system) |
| Figure | The runtime plotting object that owns axes, plot objects, labels, legends, limits, camera state, and renderable scene data. | [Figure State & Handles](/docs/runtime/plotting/state-and-handles) |
| Filesystem provider | A pluggable implementation behind path-based I/O. Providers can target local disk, browser storage, remote storage, or custom host-backed filesystems. | [Filesystem Abstraction](/docs/runtime/fs) |
| Finalizer | A cleanup hook registered with the GC for values that own external resources. GPU tensors use finalizers so provider buffers can be released when the GC-managed value is collected. | [Memory Management](/docs/runtime/gc) |
| Future | The lazy value returned by calling a RunMat async function. The function body runs when the future is awaited or spawned. | [Async Execution](/docs/runtime/execution/async) |
| Full snapshot | A complete host-facing view of workspace state. The session asks for a full snapshot when upserts/removals alone are not enough for a host to update safely. | [Workspace State](/docs/runtime/session/workspace) |
| Function registry | Session-owned state for user-defined semantic functions that persist across interactive inputs. The compiler and VM use it to resolve calls in later requests. | [Session Engine](/docs/runtime/session) |
| Fusion engine | The Accelerate component that recognizes operation chains that can stay on device and run as fused GPU work instead of materializing intermediate arrays on the host. | [Fusion Engine & Residency Management](/docs/runtime/gpu/fusion) |
| Fusion plan | A structured description of the operations, residency decisions, cache behavior, and fallback points for a candidate fused execution path. | [Fusion Engine & Residency Management](/docs/runtime/gpu/fusion) |

## G

| Term | Definition | More |
| --- | --- | --- |
| GC | Garbage collector. `runmat-gc` manages address-stable runtime values that need shared identity, root tracking, finalizers, or cyclic lifetime management across the VM and runtime. | [Memory Management](/docs/runtime/gc) |
| GC root | An entry point that keeps a GC-managed value alive. Roots include explicit handles, VM stack values, VM variables, session values, and remembered old-to-young references. | [Memory Management](/docs/runtime/gc) |
| GPU provider | The backend interface that owns GPU buffers, dispatches kernels, gathers data, reports profiling counters, and implements provider-specific acceleration hooks. | [wgpu Backend & Accelerate Provider](/docs/runtime/gpu/wgpu) |
| GPU tensor | A tensor whose data is resident on the GPU. It carries provider-owned state and is gathered only when host materialization or a host-only builtin requires it. | [GPU Acceleration & Fusion Engine](/docs/runtime/gpu) |

## H

| Term | Definition | More |
| --- | --- | --- |
| Handle | A stable runtime identity used for stateful objects such as figures, axes, surfaces, object handles, or provider-owned resources. A handle is not the same thing as the value payload it references. | [Figure State & Handles](/docs/runtime/plotting/state-and-handles) |
| HIR | High-Level Intermediate Representation. HIR is the first compiler representation after parsing and is where scopes, binding IDs, callable identities, imports, captures, and source-level structure are resolved. | [High-Level IR (HIR)](/docs/runtime/compiler/hir) |
| Host | The application embedding RunMat: CLI, REPL, browser, notebook, editor, desktop app, or server process. Hosts submit execution requests and consume structured results. | [Host Integration](/docs/runtime/session/host-integration) |
| Host policy | Request/session settings supplied by the embedding host. Host policy controls behavior such as output capture, source naming, input handling, workspace updates, and compatibility boundaries. | [Execution Requests](/docs/runtime/session/execution-requests) |
| Hover | An LSP feature that returns information for a symbol under the cursor, such as builtin documentation, variable information, function signatures, or diagnostic context. | [Editor Features](/docs/runtime/lsp/features) |

## I

| Term | Definition | More |
| --- | --- | --- |
| Index plan | The compiler representation of an indexing operation, including scalar indexing, slicing, `end` expressions, cell indexing, and deletion forms. | [Indexing Subsystem](/docs/runtime/vm/indexing) |
| Ingestion key | The key used by official builds to authenticate telemetry delivery to the hosted collector. Source builds without a key can still print payloads and use local provider counters. | [Telemetry](/docs/runtime/development/telemetry) |
| Instruction | A single VM bytecode operation, represented by `Instr` in the VM. Instructions define the stack, variable, call, indexing, control-flow, async, and runtime service operations the interpreter executes. | [Bytecode Compilation](/docs/runtime/vm/bytecode) |
| Interpreter | The VM execution tier that runs bytecode directly. It is the semantic baseline for RunMat execution and the fallback when JIT execution is unavailable or ineligible. | [VM Interpreter & Bytecode](/docs/runtime/vm) |
| IR | Intermediate representation. RunMat uses HIR and MIR to move from source-shaped syntax toward analyzable control flow and executable bytecode. | [Compilation Pipeline](/docs/runtime/compiler) |

## J

| Term | Definition | More |
| --- | --- | --- |
| JIT | Just-in-time compilation. RunMat's JIT tier compiles eligible bytecode through Turbine into native execution paths while preserving VM fallback behavior. | [JIT Compiler](/docs/runtime/jit) |

## L

| Term | Definition | More |
| --- | --- | --- |
| Lexer | The compiler component that turns source text into tokens before parsing. | [Lexer & Parser](/docs/runtime/compiler/lexer-and-parser) |
| Logical truth | MATLAB truthiness rules used by control flow and logical operations. RunMat normalizes those checks through VM/runtime helpers instead of plain Rust boolean conversion. | [Interpreter Dispatch & Execution Loop](/docs/runtime/vm/interpreter) |
| Lowering | Translation from one compiler representation to the next, such as AST to HIR or HIR to MIR. Lowering preserves source meaning while making later analysis or execution more explicit. | [Compilation Pipeline](/docs/runtime/compiler) |
| LSP | Language Server Protocol. RunMat's LSP powers editor diagnostics, highlighting, hover, completion, navigation, signature help, formatting, and symbols. | [Language Server Protocol](/docs/runtime/lsp) |

## M

| Term | Definition | More |
| --- | --- | --- |
| MAT payload | Binary workspace data used by save/load and workspace replay paths. Session replay can encode MAT bytes in a host-facing JSON envelope. | [Snapshots & Replay](/docs/runtime/session/snapshots) |
| MException | MATLAB-compatible error value used for catch/rethrow behavior and structured runtime failures. | [Errors & Diagnostics](/docs/runtime/execution/errors) |
| MIR | Mid-Level Intermediate Representation. MIR turns HIR into explicit control flow, statements, terminators, places, rvalues, and analysis inputs for bytecode generation. | [Mid-Level IR (MIR)](/docs/runtime/compiler/mir) |
| MIR analysis | Static analysis over MIR. It computes facts used by diagnostics, bytecode compilation, async behavior, spawn safety, and future optimization work. | [MIR & Static Analysis](/docs/runtime/compiler/static-analysis) |

## O

| Term | Definition | More |
| --- | --- | --- |
| Object dispatch | Runtime resolution of MATLAB object behavior such as methods, property access, `subsref`, `subsasgn`, getters, and setters. | [Callable Resolution & Function Dispatch](/docs/runtime/vm/dispatch) |
| Opcode | The operation kind encoded by a bytecode instruction. In the docs, opcode and instruction are often used together when discussing VM execution behavior. | [Bytecode Compilation](/docs/runtime/vm/bytecode) |

## P

| Term | Definition | More |
| --- | --- | --- |
| Parser | The compiler component that turns tokens into an AST and reports syntax diagnostics. | [Lexer & Parser](/docs/runtime/compiler/lexer-and-parser) |
| Place | A MIR location that can be read from or written to, such as a local, binding, field, or indexed target. Places let assignment and mutation paths remain explicit. | [Mid-Level IR (MIR)](/docs/runtime/compiler/mir) |
| Preview token | A short-lived selector returned in workspace snapshots so hosts can request bounded materialization without holding direct references to runtime values. | [Variable Inspection](/docs/runtime/session/variable-inspection) |
| Profiling | Per-request timing and provider data returned to hosts or benchmark tools. Profiling can include wall time, CPU/GPU timing, provider counters, and fusion metadata. | [Execution](/docs/runtime/execution) |
| Provider telemetry | Local counters reported by acceleration providers, such as GPU dispatches, upload/download bytes, cache hits, and fallback counts. These counters are also useful for benchmarks. | [Telemetry](/docs/runtime/development/telemetry) |

## R

| Term | Definition | More |
| --- | --- | --- |
| REPL | Read-eval-print loop. The CLI REPL keeps one `RunMatSession` alive and submits each entered line as an execution request. | [Host Integration](/docs/runtime/session/host-integration) |
| Remote I/O | Filesystem work backed by a remote provider instead of local disk. The provider interface allows remote reads and writes to be scheduled in parallel so throughput can saturate the network when storage allows it. | [Filesystem Abstraction](/docs/runtime/fs) |
| Replay | Reconstructing saved runtime state. RunMat uses replay for workspace variables and plotting scenes, but startup snapshots are a separate mechanism. | [Snapshots & Replay](/docs/runtime/session/snapshots) |
| Residency | Whether a value is currently host-resident, GPU-resident, or able to stay on device across operations. Fusion and provider paths use residency to avoid unnecessary transfers. | [Fusion Engine & Residency Management](/docs/runtime/gpu/fusion) |
| Runtime | The execution support layer below the session and VM. It owns builtins, values, warnings, console streams, plotting hooks, input hooks, object helpers, and provider integration. | [Execution](/docs/runtime/execution) |
| Rvalue | A MIR expression that produces a value for an assignment or temporary. Rvalues make computation explicit before bytecode generation. | [Mid-Level IR (MIR)](/docs/runtime/compiler/mir) |

## S

| Term | Definition | More |
| --- | --- | --- |
| Semantic function | A function known to the compiler/runtime as a callable semantic entity with compiler-visible identity. Semantic functions support user-defined functions, closures, async calls, and interactive function registry updates. | [Callable Resolution & Function Dispatch](/docs/runtime/vm/dispatch) |
| Semantic tokens | LSP classification data used by editors for syntax-aware highlighting. RunMat emits tokens from parser and compiler context instead of relying only on text patterns. | [Diagnostics & Highlighting](/docs/runtime/lsp/diagnostics-and-highlighting) |
| Session | The long-lived execution object that connects source compilation, VM/JIT execution, workspace state, host policy, plotting, diagnostics, telemetry, and result assembly. | [Session Engine](/docs/runtime/session) |
| Slice | A range-like indexing selector over one or more dimensions. Slices are represented explicitly so CPU, GPU, dataset, and remote filesystem paths can avoid materializing unnecessary data. | [Indexing Subsystem](/docs/runtime/vm/indexing) |
| Snapshot | A serialized payload used for startup acceleration or state transfer. Startup snapshots package standard-library metadata and caches; workspace replay snapshots preserve live variables. | [Snapshots & Replay](/docs/runtime/session/snapshots) |
| Source identity | Stable metadata attached to submitted source, such as a file path, REPL name, notebook cell name, or host-provided label. Diagnostics and workspace keys use it to stay tied to the right source. | [Execution Requests](/docs/runtime/session/execution-requests) |
| Spawn handle | The single-use value returned by `spawn(future)`. In the current runtime, spawning resolves the future before returning the handle; it does not schedule background work yet. | [Async Execution](/docs/runtime/execution/async) |
| Static analysis | Compile-time reasoning over source or IR. RunMat uses it for diagnostics, assignment checks, type/shape facts, async/spawn metadata, and later execution decisions. | [MIR & Static Analysis](/docs/runtime/compiler/static-analysis) |
| Startup snapshot | A binary payload that packages standard-library metadata and caches to reduce startup cost. It is separate from workspace replay. | [Snapshots & Replay](/docs/runtime/session/snapshots) |
| Surface | A host presentation target for plotting. A figure can outlive a surface, and a surface can be rebound to another figure. | [Plotting Host Integration](/docs/runtime/plotting/host-integration) |

## T

| Term | Definition | More |
| --- | --- | --- |
| Telemetry | Bounded runtime analytics and provider counters used to understand installation health, execution behavior, acceleration usage, failures, and benchmark characteristics. | [Telemetry](/docs/runtime/development/telemetry) |
| Terminator | The MIR instruction that ends a basic block, such as branch, return, jump, or await. Terminators make control flow explicit. | [Mid-Level IR (MIR)](/docs/runtime/compiler/mir) |
| Turbine | RunMat's JIT compiler crate. It compiles eligible bytecode paths into native code and falls back to the VM when a path cannot be compiled safely. | [JIT Compiler](/docs/runtime/jit) |
| TypeScript bindings | The `bindings/ts` package that exposes RunMat's WASM runtime, session API, LSP bundle, startup snapshot, plotting hooks, and host integration types to JavaScript and TypeScript callers. | [WASM & TypeScript/JavaScript](/docs/runtime/wasm) |

## V

| Term | Definition | More |
| --- | --- | --- |
| Value | The concrete runtime representation used for values produced, stored, and passed around during RunMat execution, including tensors, logical arrays, strings, cells, structs, objects, handles, closures, and GPU tensors. | [Runtime Values & Type Model](/docs/runtime/values) |
| Variable array | VM slot storage for the active execution. The session prepares it from durable workspace values before execution and harvests it back afterward. | [Workspace State](/docs/runtime/session/workspace) |
| VM | Virtual machine. The VM compiles MIR into bytecode, executes bytecode in the interpreter, performs call and indexing dispatch, and provides the semantic baseline for execution. | [VM Interpreter & Bytecode](/docs/runtime/vm) |

## W

| Term | Definition | More |
| --- | --- | --- |
| WASM | WebAssembly. RunMat's WASM package exposes the runtime and LSP to browser and JavaScript hosts. | [WASM & TypeScript/JavaScript](/docs/runtime/wasm) |
| WebGPU | Browser GPU API used by WASM hosts when GPU support is available. It maps to the same high-level acceleration model as native provider-backed execution. | [WASM & TypeScript/JavaScript](/docs/runtime/wasm) |
| WGPU | Rust graphics and compute abstraction used by RunMat's current native and web acceleration backend. | [wgpu Backend & Accelerate Provider](/docs/runtime/gpu/wgpu) |
| Workspace | The session's live variable state. It bridges durable host-visible values and VM slots during execution. | [Workspace State](/docs/runtime/session/workspace) |
| Workspace delta | Versioned upserts, removals, and full-snapshot requests emitted after execution so hosts can update variable panes without rebuilding them blindly. | [Workspace State](/docs/runtime/session/workspace) |
| Workspace handle | Stable request/session identity used to build interactive binding keys and associate source execution with a particular workspace. | [Execution Requests](/docs/runtime/session/execution-requests) |
| Workspace replay | Exporting and importing live workspace variables through a bounded payload. Replay replaces the current workspace with restored variables; it does not merge into the existing workspace. | [Snapshots & Replay](/docs/runtime/session/snapshots) |
| Write barrier | GC bookkeeping used when an older object is updated to reference a younger object. It keeps minor collections correct by adding remembered-set roots. | [Memory Management](/docs/runtime/gc) |
