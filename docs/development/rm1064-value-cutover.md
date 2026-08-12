# RM-1064 value structural cutover baseline

This document closes the human-readable portion of I03C. The machine-readable companion is `rm1064-value-cutover.json`, and `scripts/development/check-rm1064-value-cutover.mjs` verifies that its declared source ownership remains exact. The baseline is signed integer commit `f24908ddc2296dc5640a0a15e3cbb8ff9f7dee53`, qualified and adopted through true merge `2e7bd5ed1543d57872c8be9426b4d09ab5feac0d`. R03 may relocate this model into `runmat-value`; it may not change the semantics below or claim that the independently continuing integer epic is complete.

## Authority and sequencing

The current live authority is `runmat-builtins`; this is temporary physical ownership, not the target domain boundary. R03 atomically creates `runmat-value`, moves the complete live-value closure, migrates all consumers, deletes the old definitions, and updates the machine-readable manifest to the extracted stage. R04–R20 then consume only public typed contracts. Once R03 begins, later intermediate integer publications are audit-only. I04 after R20 preserves the terminal branch as true ancestry and ports every `f24908ddc..terminal` semantic change through the relocation manifest into the current layout.

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
- `ObjectInstance` currently closes over `DynamicPropertyDef` and `Access`. R03 moves that closure without duplicating it; R04 immediately separates immutable declaration vocabulary into `runmat-types` and session/live dynamic-property state into the appropriate value/runtime modules. This temporary ownership is recorded, not exposed as a stable target API.

### Symbolic values and presentation

- `SymbolicExpr` and `SymbolicFunction` are intrinsic value representation and move to `runmat-value`; `SymbolicDeclaration`, its parser/error, and tokenization are `syms` command behavior and move to runtime.
- Session display format and mutable formatting preference do not move into `runmat-value`. Intrinsic value inspection stays dependency-light; runtime presentation owns user-configured formatting. R03 must migrate call sites coherently because Rust orphan rules preclude implementing `Display` for a foreign value from runtime.
- Static `Type`, literal/resolve context, builtin descriptors, function pointers, class declarations, and mutable class/static-value registries are not part of the live value. Their exact R04–R09 destinations are listed in the machine-readable manifest.

## Target crate and dependency boundary

`runmat-value` uses the modular layout specified by `RM_1064_REMAINING_SCOPE_PLAN.md`: focused `numeric`, `array`, `aggregate`, `symbolic`, `object`, and `callable` modules plus execution, exception, trace, and intrinsic inspection modules. `lib.rs` and `value.rs` remain small composition facades. The crate may depend on dependency-light handle/API crates such as `runmat-accelerate-api`, `runmat-execution`, and `runmat-gc-api`; it may not depend on runtime, HIR, MIR, VM, native codegen, filesystem/network/process facilities, provider implementations, session registries, or presentation policy.

After R03, dependency guards prove that `runmat-builtins` neither defines nor re-exports live value/storage symbols. A direct, explicitly temporary dependency on `runmat-value` remains only because the pre-R06 `BuiltinFunction`/constant registry, `Type::from_value`, and pre-R04 class defaults still embed live values. R04/R06 relocate those consumers, update the manifest to `catalog-separated`, and remove the dependency completely. Only at that gate are static catalog declarations required to be usable by HIR/LSP/WASM without linking live values or executable runtime behavior.

## Pre-I04 architecture envelope

R04–R20 may define shared identities/facts, catalog/runtime joins, semantic services, executable schemas, Native IR, JIT/AOT, placement, reachability, and internal artifacts. Those systems must be class-parametric and capability-driven. They may not match private storage variants outside `runmat-value`, embed Rust layout/size/alignment into a public boundary, promise zero-copy compatibility, or claim terminal integer completeness.

The representative pilot identities are `zeros`, `full`, `abs`, `gather`, `struct`, and `feval`. Together they exercise dense exact integer/class/shape/provider construction, sparse storage, complex values, opaque provider materialization/provenance, recursive aggregate/GC/codec behavior, and dynamic callable/effect/slow-path behavior. Each pilot cuts over atomically when its owning slice arrives. These pilots do not increment exhaustive A–G cohort completion; systematic C00–C07 waits for I04.

## Verification contract

I03C inherits the complete qualified Slice 417 evidence recorded in the root progress ledger: affected OCCT-free checks; accelerate API; constructor, GPU, dispatcher, and compatibility runtime suites; VM and LSP integration; residency/stochastic tests; selected local WGPU hardware gates; strict affected-crate Clippy; formatting/diff/conflict guards; standard WASM registry regeneration; and RM-1064 inventory freshness. R03 reruns this matrix against `runmat-value` and adds architecture/dependency, native/WASM codec, GC, aggregate, object, and presentation-boundary tests.

OCCT remains off for I03C/R03 unless the actual extraction exposes a geometry/OCCT dependency. Linux and Windows validation remains assigned to R31.
