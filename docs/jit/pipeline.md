# JIT Compilation Pipeline

The JIT consumes the same immutable executable unit already produced for a request. It does not reparse source, reconstruct control flow from bytecode, or define a second runtime value representation.

## Compile And Publication Flow

```mermaid
flowchart TD
  Unit["ExecutableUnit<br/>MIR, analysis, bytecode, source map"]
  Lower["runmat-native-codegen<br/>MIR to Native IR"]
  Verify["schema and semantic verification"]
  CLIF["Native IR to Cranelift IR"]
  Code["machine code + measured metadata"]
  Admit["dependency, profile, version,<br/>resource, and byte-budget admission"]
  Cell["stable published entry cell"]
  Host["runmat-native-executor<br/>invocation host"]
  Runtime["runmat-runtime semantics"]

  Unit --> Lower --> Verify --> CLIF --> Code --> Admit --> Cell --> Host --> Runtime
```

`runmat-native-codegen` owns portable Native IR and machine-code lowering. `runmat-jit` owns adaptive compilation, tiering, publication, invalidation, and dynamically allocated code ownership. `runmat-native-executor` owns entrypoint binding, invocation state, roots, deoptimization, OSR, and semantic host callbacks; linked AOT and JIT entrypoints use that same host without making the AOT runtime depend on JIT policy. `runmat-core` owns session selection, immutable executable products, dependency revisions, and host-visible result/workspace commitment. `runmat-runtime` remains executor-neutral and owns language operations and the versioned native ABI.

## Native Invocation

Generated code receives opaque, generation-checked value references and a versioned host table. Runtime `Value` layout never crosses the native ABI. The invocation host retains locals, GC roots, exact source and resume identity, loop iterators, exception handlers, outputs, cancellation state, and any transactional workspace snapshot.

Semantic sites call the same runtime operations used by the VM. Structured exits distinguish completion, exception, cancellation, suspension, and deoptimization. A deoptimization frame includes its exact function, block, position, phase, ordinal, live locals, roots, aliases, and side-effect epoch so continuation does not replay completed work.

The Cranelift lowerer emits a host-site call for each Native IR instruction and terminator, then branches on the validated outcome. Generic JIT versions, specialized JIT versions and linked AOT objects use this lowerer. Representation guards select versions; Runtime numeric-region plans select the arithmetic implementation.

## Tiering And Specialization

Feedback is keyed by stable entry, function, and optional loop-header identity. Exact runtime facts form bounded representation profiles. Policy admits generic native code after generic heat, and guarded optimized versions only after their matching profile becomes hot. Compilation failures retain the existing continuation; runtime program errors are not mistaken for compiler failures.

Every published target records exact program, builtin-catalog, project, and referenced session-function generations. Dynamic lookup also depends on the session catalog. Publication rechecks those generations after background compilation. Redefining a referenced function retires stale code even if the function keeps the same semantic ID.

## Optimized Regions And Placement

Specialized products may include verified effect-free numeric regions. The JIT derives the region plan, and Runtime retains its validated numeric DAG and operation identities. The session's shared placement authority considers the eligible implementations: original native execution, vectorized CPU execution, CPU evaluation through Runtime's original operations, and provider fusion. Each route must support the input types and shapes and supply its required resource bounds and completion capabilities. The JIT does not maintain separate GPU policy.

The `Vectorized` CPU route evaluates the admitted scalar/dense-double DAG in one element traversal through Runtime's compiled Rust kernel. A host-site checkpoint selects the route; after it publishes the results, native execution skips the covered sites. Other representations continue through their ordinary Runtime operations. Cancellation and errors retain the invocation's originating source and continuation state.

Placement compares complete routes, including preparation, transfers, and synchronization. Estimates and observations remain tied to the implementation and execution boundary they describe. A missing complete-route price stays unknown; bounded exploration can measure an otherwise legal route when the selected runtime supports it. Explicit `gpuArray` intent excludes host alternatives. Rejection before effects can retain ordinary execution; a failed or cancelled execution is not replayed through another route.

## Interactive Workspace Contract

An eligible interactive input is compiled from the MIR and analysis already produced for that request. Its native invocation receives all ambient assigned values plus exact semantic bindings for locals used by the source. Dynamic builtins such as `exist`, `which`, `save`, `load`, `clear`, and `clearvars` therefore observe the invocation workspace through the common runtime service.

Local writes and runtime workspace effects update invocation-owned state. Only successful completion returns a snapshot to Core. Core then applies the existing display, semicolon suppression, `ans`, workspace-delta, and function-publication policies. An exception or cancellation cannot leak a partial snapshot and is never followed by whole-request VM replay.

## WebAssembly

The portable executable manifest and Native IR schema are target-neutral and validated in WASM builds. The Cranelift JIT module and executable-memory lifecycle are native-only. Browser execution keeps the VM/platform executor as its semantic path while sharing compiler facts, runtime services, portable identities, and artifact validation.
