# RM-1064 value structural cutover baseline

This document records the value cutover from its I03C structural baseline through the I04 terminal semantic seal. The machine-readable structural companion is `rm1064-value-cutover.json`, and `scripts/development/check-rm1064-value-cutover.mjs` verifies that its declared source ownership remains exact. The structural baseline is signed integer commit `f24908ddc2296dc5640a0a15e3cbb8ff9f7dee53`, qualified and adopted through true merge `2e7bd5ed1543d57872c8be9426b4d09ab5feac0d`. The terminal integer result is signed commit `25ab190277c241806ad92ff53930f44eb6e96c36`; I04 adopts it as true ancestry and reconciles its semantics into the modular ownership described here.

## Authority and sequencing

The live authority is now `runmat-value`. R03 created the crate, moved the complete live-value closure, migrated all consumers, deleted the old definitions, and advanced the machine-readable manifest to the extracted stage. R04–R20 consume only public typed contracts. Later intermediate integer publications are audit-only. I04 after R20 preserves the terminal branch as true ancestry and ports every `f24908ddc..terminal` semantic change through the relocation manifest into the current layout.

The terminal merge may not restore an old path merely to make a textual merge easy. Every terminal-range path and semantic hunk must be classified as ported, already superseded equivalently, regenerated from its current authority, or intentionally inapplicable. An unclassified hunk, duplicate implementation, compatibility re-export, resurrected `runmat-builtins::Value`, or stale generated product fails I04.

## Baseline semantic invariants

### Numeric classes and scalars

- Real numeric storage has ten exhaustive physical classes: `F64`, `F32`, signed `I8/I16/I32/I64`, and unsigned `U8/U16/U32/U64`.
- `F32` is physically stored as `f32`; integer values are physically stored in their native width. Neither is an `f64` label over widened storage.
- `IntValue`, `NumericScalar`, `IntegerStorage`, and `NumericStorage` preserve exact values. Wide `u64` and `i64` values never pass through `f64` in an exact path.
- There is no universal borrowed floating view. `as_f64_slice` and `as_f32_slice` succeed only for matching physical storage; `materialize_f64` and `materialize_f32` are explicit conversion boundaries whose possible precision loss is visible in the API name.
- Integer casts, assignment, zero/one allocation, gather/reorder, comparison, and class derivation dispatch exhaustively over native storage. Adding a class must force compiler-visible coverage.

### Dense, sparse, and complex arrays

- `Tensor` owns one private authoritative `TensorStorage` plus MATLAB-visible column-major shape. Compatibility `rows`/`cols` caches derive from shape and do not become a second shape authority.
- Construction validates element count with overflow-aware shape arithmetic. Structural operations use authoritative storage length and typed access, never a stale or lossy mirror.
- `SparseTensor` is two-dimensional CSC with validated `col_ptrs`, sorted row indices, and private `F64`, `F32`, exact integer, or logical value storage. Sparse integer values remain exact and sparse logical has no fabricated numeric payload.
- `ComplexTensor` owns private `F64`, `F32`, or exact integer complex storage. Exact complex integer real/imaginary components have the same class and length; their sign, width, and wide values remain exact.
- Logical, character, string, and symbolic arrays retain their current shape/layout contracts. Character and cell payload ordering differences are explicit at column-major conversion boundaries rather than silently normalized.

### Live value categories

- `Value` remains the single live recursive runtime sum type. It covers exact/scalar numeric values, logical/text/numeric/symbolic arrays, cells, structs, objects, opaque GPU handles, callable handles/closures, exceptions, output lists, and execution handles.
- `runmat-execution::ValuePayload` remains inert, bounded transport data. It never enters VM stacks, gains runtime behavior, or becomes a second live value model.
- GPU values carry opaque `GpuTensorHandle` identities. Provider ownership, `(device_id, buffer_id)` identity, physical precision/storage metadata, and explicit-versus-automatic provenance remain owned by acceleration contracts; `runmat-value` does not become a placement planner or provider registry.
- Future, task, pool, and job variants wrap stable `runmat-execution` handles only. Scheduling, cancellation, transport, and lifecycle policy remain in execution services.
- Function handles and closures preserve their current external, method, bound-function, and capture identities. Resolution and invocation remain runtime/compiler responsibilities.

### Aggregates, objects, GC, and errors

- Cells, structs, object arrays, object instances, closures, and output lists recursively own live `Value` payloads. Their construction and traversal preserve shape, field order where defined, and GC reachability.
- `Trace` implementations move with the values and depend only on `runmat-gc-api`. The collector, roots, session, and allocation policy do not move into `runmat-value`.
- `HandleRef` and `Listener` preserve GC identity and validity semantics. Object equality does not degrade into structural equality for handle objects.
- `MException` remains a live exception payload. Runtime error creation, stack policy, throwing, catching, and presentation remain outside the value crate.
- `ObjectInstance` retains only its live `DynamicPropertyDef` payload. R04 removed the temporary `Access` declaration and made dynamic properties consume `runmat-types::MemberAccess` directly, without an alias or re-export. Immutable class/property/method declarations now live in `runmat-types`; executable class registrations and static values live in `runmat-runtime`.

### Symbolic values and presentation

- `SymbolicExpr` and `SymbolicFunction` are intrinsic value representation and live in `runmat-value`. `SymbolicDeclaration`, its parser/error, and tokenization are shared declaration syntax used by both HIR lowering and runtime `syms` execution; R04 moves that vocabulary into dependency-neutral `runmat-types` rather than introducing a HIR-to-runtime dependency.
- Session display format and mutable formatting preference ultimately belong to runtime presentation. R03 temporarily colocates the existing `Display` implementation and its format policy in `runmat-value`, because Rust orphan rules prohibit leaving that implementation in `runmat-builtins` or moving it onto a foreign type in runtime. This is an explicit behavior-preserving accommodation, not target ownership: R09 replaces it with runtime-owned presentation and dependency-light intrinsic inspection.
- Static `Type`, literal/resolve context, builtin descriptors, function pointers, class declarations, and mutable class/static-value registries are not part of the live value. Their exact R04–R09 destinations are listed in the machine-readable manifest.

## Target crate and dependency boundary

`runmat-value` uses the modular layout specified by `RM_1064_REMAINING_SCOPE_PLAN.md`: focused `numeric`, `array`, `aggregate`, `symbolic`, `object`, and `callable` modules plus execution, exception, trace, and intrinsic inspection modules. `lib.rs` and `value.rs` remain small composition facades. The crate may depend on dependency-light handle/API crates such as `runmat-accelerate-api`, `runmat-execution`, and `runmat-gc-api`; it may not depend on runtime, HIR, MIR, VM, native codegen, filesystem/network/process facilities, provider implementations, session registries, or presentation policy.

After R03, dependency guards prove that `runmat-builtins` neither defines nor re-exports live value/storage symbols. R04 additionally proves that `runmat-runtime` does not depend on HIR, HIR does not depend on runtime or live values directly, and `runmat-types` has no RunMat crate dependency. A direct, explicitly temporary `runmat-builtins -> runmat-value` dependency remains only because the pre-R06 executable builtin/constant registry and `Type::from_value` still embed live values. R06 relocates those consumers, advances the manifest to `catalog-separated`, and removes the dependency completely. Only at that gate are all static catalog declarations required to be usable by HIR/LSP/WASM without linking live values or executable runtime behavior.

## R04 shared semantic boundary

`runmat-types` is the dependency-neutral owner of stable local/source/callable/class/member identities, immutable serializable declarations, symbolic declaration syntax, and the recursive `ValueFact` algebra. The fact model records numeric class/domain, rank and dimensions, dense/sparse/opaque storage, logical layout, contiguity, materialized/view state, host/device/remote residency, alias and mutation behavior, aggregate/object/callable/foreign/execution structure, certainty/dynamic reasons, and invalidation causes. Heterogeneous cells retain position-preserving recursive facts as well as a conservative arbitrary-element summary.

Live-value adaptation belongs to `runmat-runtime::value_fact`; `runmat-types` never imports or inspects `Value`. The adapter exhaustively matches the current `Value` enum. Provider-resident values preserve exact numeric class and real/complex domain when acceleration metadata proves it; a handle missing element metadata remains an explicit `UnsupportedRepresentation` dynamic boundary rather than being guessed as `double`. Canonical schema-versioned JSON vectors and representative finite lattice-law tests run on native and compile for WASM.

Mutable class state is now runtime-owned. `RuntimeClass`, `RuntimeProperty`, and `RuntimeMethod` are deliberately named registrations/bindings rather than declarations: they may contain live defaults, executable names, and session static values. The GC discovers runtime-owned static roots through a dependency-inverted external-root provider, so the collector no longer depends on builtin or runtime class state. Immutable standard-class declarations exposed to HIR are separate `runmat-types` values and never consult the session registry.

## Pre-I04 architecture envelope

R04–R20 may define shared identities/facts, catalog/runtime joins, semantic services, executable schemas, Native IR, JIT/AOT, placement, reachability, and internal artifacts. Those systems must be class-parametric and capability-driven. They may not match private storage variants outside `runmat-value`, embed Rust layout/size/alignment into a public boundary, promise zero-copy compatibility, or claim terminal integer completeness.

The representative pilot identities are `zeros`, `full`, `abs`, `gather`, `struct`, and `feval`. Together they exercise dense exact integer/class/shape/provider construction, sparse storage, complex values, opaque provider materialization/provenance, recursive aggregate/GC/codec behavior, and dynamic callable/effect/slow-path behavior. Each pilot cuts over atomically when its owning slice arrives. These pilots do not increment exhaustive A–G cohort completion; systematic C00–C07 waits for I04.

## Verification contract

I03C inherits the complete qualified Slice 417 evidence recorded in the root progress ledger: affected OCCT-free checks; accelerate API; constructor, GPU, dispatcher, and compatibility runtime suites; VM and LSP integration; residency/stochastic tests; selected local WGPU hardware gates; strict affected-crate Clippy; formatting/diff/conflict guards; standard WASM registry regeneration; and RM-1064 inventory freshness. R03 reruns this matrix against `runmat-value` and adds architecture/dependency, native/WASM codec, GC, aggregate, object, and presentation-boundary tests.

OCCT remains off for I03C/R03 unless the actual extraction exposes a geometry/OCCT dependency. Linux and Windows validation remains assigned to R31.

## I04 terminal semantic seal

I04 preserves the terminal workstream's complete history while keeping the post-R20 domain boundaries authoritative. `runmat-value` remains the sole owner of `Value`, numeric scalars, dense/sparse/complex storage, aggregates, object payloads, callable values, and intrinsic value inspection. `runmat-types` owns dependency-neutral integer literal parsing and static facts. `runmat-builtins` owns static catalog declarations only. Runtime conversion, validation, callable binding, class registration, and MATLAB-visible execution semantics remain in `runmat-runtime`; provider implementation and residency policy remain outside the value crate.

The terminal merge does not expose physical storage as a public ABI. Exact integer scalars and arrays are accessed through typed, class-preserving APIs; exact serialization never routes `i64` or `u64` through `f64`; numeric promotion and MATLAB conversion policy remain explicit at their semantic boundary. Provider-resident values retain opaque handles and descriptor metadata, and automatic gathering is an execution retry policy rather than a second host-side representation. Static analysis therefore reasons from class, shape, domain, storage, residency, and certainty facts without depending on Rust layout or runtime function pointers.

`docs/development/rm1064-i04-terminal-reconciliation.json` is the durable range inventory. Its generator pins the structural source, terminal source, and post-R20 destination parent, records every terminal-range path and every merge-conflict hunk, and permits only the four I04 dispositions. The committed report must contain zero unclassified paths; its check script must reproduce the report in a fresh checkout without a temporary merge-conflict file. Two obsolete Turbine-only files are absent from the final tree because R18 removed that execution architecture before I04.

After the signed two-parent I04 merge, the pre-extraction tree is permanently historical. Later numeric or storage corrections are ordinary changes against `runmat-value` and its current consumers; they may not reintroduce the old owner, an alias, a compatibility re-export, a parallel codec, or a layout-dependent public contract. Layout-sensitive extension and zero-copy commitments remain gated on their own later design and conformance work.
