# Plan 3: Restore Runtime, Tooling, And Workspace Consumers

## Objective

Restore the workspace to a fully compiling and functioning state by migrating VM, runtime, workspace, LSP, snapshots, and JIT/Turbine consumers onto semantic HIR, MIR, and the new analysis products.

Plans 0-2 may leave broad downstream breakage. Plan 3 brings the repo back to green on the new compiler architecture.

## Desired Resting State

The repo is green on the new compiler pipeline:

- semantic HIR is the source model
- MIR is the analysis/codegen planning model
- VM layout is derived from semantic bindings and MIR locals
- workspace export is semantic
- LSP consumes semantic IDs, diagnostics, and analysis facts
- async-capable execution paths are explicit and preserved
- old `HirProgram` / `VarId` / statement-function architecture is removed from active consumers

## Core Invariants

- Runtime slots are derived layout, not semantic identity.
- VM bytecode is derived from MIR plus semantic layout maps.
- Function identity is `FunctionId` internally, with names retained for diagnostics/dynamic behavior.
- Workspace visibility comes from `HirBinding` metadata.
- Async suspension boundaries are preserved through lowering and execution.
- LSP, CLI, and desktop diagnostics consume the shared diagnostic model.

## Migration Order

1. VM layout and bytecode lowering
2. Core session execution and workspace export
3. Async host/runtime integration
4. LSP and static-analysis consumers
5. Snapshot and Turbine/JIT consumers
6. Compatibility cleanup

## Primary Crates

- `runmat-vm`
- `runmat-core`
- `runmat-lsp`
- `runmat-snapshot`
- `runmat-turbine`

## Secondary Crates

- `runmat-hir`
- `runmat-mir`
- `runmat-static-analysis`
- `runmat-runtime`
- `runmat-async`

## Implementation Plan

1. Add VM layout derivation.

Create a layout product that maps:

- `FunctionId -> UserFunction`
- semantic `BindingId -> frame/local slot` where applicable
- `FunctionAbi -> VM frame ABI` including fixed/variadic inputs, fixed/variadic outputs, and implicit `nargin`/`nargout`
- MIR temporaries to VM locals
- entrypoint policy plus workspace-visible bindings to exportable frame slots
- module/global/persistent bindings to runtime storage handles
- capture bindings to closure/environment layout

2. Compile bytecode from MIR.

Bytecode compilation should start from selected `HirEntrypoint.target`, then use the corresponding `MirBody`, entrypoint policy, and layout metadata.

3. Compile calls through semantic callee identity.

Call lowering should use:

- `FunctionId` for direct user calls
- stable builtin IDs and behavior metadata for builtins
- class constructor IDs where available
- requested-output counts for all calls
- output-list return protocol for multi-output calls
- function handle call targets and dynamic callable fallback
- comma-list expansion at call argument and assignment boundaries
- dynamic expression calls for dynamic behavior
- unresolved calls as conservative runtime lookup only where necessary

4. Preserve async execution boundaries.

VM/runtime lowering should preserve whether a call may suspend or requires async runtime support. Synchronous wrappers may exist, but async behavior should not be hidden from the compiler/runtime model.

The VM/runtime user-facing async model should match the target Rust-like semantics:

- async function calls and async blocks allocate lazy future values
- futures begin executing only when polled by `await` or scheduled by `spawn`
- `await` polls the future/task, suspends the current execution when pending, and resumes at the MIR/bytecode resume point
- `spawn` schedules a future for concurrent execution and returns a task handle
- `spawn` must reject or never receive non-spawn-safe futures from compiler lowering
- spawned task frames must not alias mutable parent frame storage
- language-synchronous builtins continue to return ordinary values to sequential code, even when their implementation internally awaits host/provider work
- cancellation/drop releases future/task frame roots and provider resources cooperatively

5. Restore core session execution.

Core execution should support:

- REPL/snippet execution
- function entrypoint execution
- script-like entrypoint execution
- async host interaction
- compatibility file execution where still supported

Core execution must implement the MATLAB call and mutation ABI:

- pass actual input count and requested output count to user functions and builtins
- pack excess inputs into `varargin`
- expose `nargin` and `nargout` inside function frames
- return output lists and destructure them into assignment targets
- support discarded outputs
- expand comma-separated lists in function arguments and assignments
- implement indexed assignment growth, deletion by `[]`, struct field creation, scalar expansion, and slice assignment shape checks
- execute function handles, `feval`, dynamic callable fallback, and anonymous function captures through the same call ABI
- execute empty-array concat/deletion roles, scalar/implicit expansion, matrix/elementwise operators, string/char distinctions, struct/cell/object aggregate behavior, and core control-flow semantics at the minimum level needed for representative compatibility tests
- execute workspace effects from `load`, `clear`, `global`, `persistent`, `eval`, `evalin`, and `assignin` through explicit session/workspace APIs
- execute or explicitly diagnose environment effects from path/cwd/cache mutation APIs through explicit session/environment APIs

6. Implement semantic workspace export.

After execution, export workspace-visible values using:

- selected entrypoint
- `HirAssembly`
- VM layout
- final frame/storage values
- entrypoint workspace export policy
- binding visibility metadata

Rules:

- `TopLevel` bindings may appear after script-like or REPL execution
- `ModuleVisible` bindings may appear according to host policy
- `ImplicitAns` appears when implicit result behavior produces it
- `Hidden` bindings never appear in ordinary workspace views

7. Restore LSP on semantic products.

LSP should use:

- modules for source organization
- functions/classes/bindings for symbols
- resolver products for navigation
- `ExprId` and `BindingId` facts for hover/diagnostics
- structured diagnostics
- class metadata where available
- call arity, comma-list, indexing, assignment, workspace/environment-effect, and class-dispatch facts where available

8. Restore snapshots.

Snapshots should preserve semantic source identity and enough function/module/binding/layout information to reload or inspect compiled state without reviving old `HirProgram` semantics.

9. Restore Turbine/JIT integration.

Turbine should consume the VM bytecode product and semantic layout metadata. It should not depend on old HIR statement forms.

10. Clean up compatibility shims.

Remove:

- legacy remapping helpers based on `VarId`
- string-keyed function reconstruction helpers
- old inference maps keyed by function name
- old HIR tests that assert legacy statement forms
- transitional compatibility views introduced only for migration

## Tests

Add or restore tests for:

- simple script execution
- function execution
- multi-output destructuring and requested-output-sensitive builtins such as `size` and `max`
- `nargin`, `nargout`, `varargin`, and `varargout` execution
- comma-list expansion from cells and `varargin{:}`
- indexed assignment growth, deletion, and symbolic `end` slice assignment
- command syntax lowered to normal execution calls
- workspace/environment effects for `load`, `clear`, path, and cwd operations where supported
- function handles and `feval`
- empty-array concat/deletion, scalar/implicit expansion, and operator execution
- struct/cell aggregate behavior and string/char basics
- `for` column iteration, scalar conditions, short-circuit, and `switch/case` basics
- nested function capture execution
- `isolated` rejection
- anonymous function execution
- class method execution where supported
- workspace export for scripts and REPL snippets
- function locals not leaking into workspace
- async input/plotting/provider paths still work
- async function/block calls are lazy until awaited or spawned
- `spawn` creates task handles and explicit concurrency
- `spawn` does not allow mutable lexical capture races
- live values across await remain rooted and are released on completion/cancellation
- LSP symbol/hover diagnostics on new IDs/facts
- snapshot roundtrip
- JIT fallback behavior on unsupported semantic instructions

## Acceptance Criteria

- The workspace builds.
- Core execution works through semantic HIR -> MIR -> VM lowering.
- Workspace snapshots are built from semantic binding visibility.
- VM execution implements the Function ABI, requested-output call protocol, comma-list expansion, and place mutation semantics.
- Async-capable paths remain supported and explicit.
- LSP builds against semantic IDs, diagnostics, and analysis facts.
- Snapshot and Turbine/JIT crates build or have clearly documented unsupported cases.
- Old `HirProgram`-centered architecture is removed from active compiler/runtime paths.

## Explicit Non-Goals

- Do not complete full MATLAB core semantics; that is Plan 4.
- Do not implement project manifests and package composition; that is Plan 5.
- Do not implement runtime class/builtin metadata generation; that is Plan 6.
- Do not complete accelerate/fusion/GC lifetime hardening; that is Plan 7.
- Do not remove every runtime string name; display names, builtin names, diagnostics, and dynamic lookup still need strings.
