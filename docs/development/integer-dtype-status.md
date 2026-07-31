---
title: "Integer dtype compatibility status"
category: "Development"
section: "14.6"
last_updated: "July 31, 2026"
---

# Integer dtype compatibility status

This document is the durable status and recovery ledger for the first-class integer work on `end-july-work`. The work tracks eval-loop issue `ri-b4e5076574b9`.

The branch forks from the integer commit that was subsequently merged through `dev` and `main` (`66ead7582`). It intentionally does not include later `dev` or `main` changes.

The completion program is controlled by the external operator artifacts `../MATLAB_INTEGER_PLAN.md` and `../MATLAB_INTEGER_PROGRESS.md`. The durable compiler/callsite disposition format is [`integer-storage-migration-ledger.md`](./integer-storage-migration-ledger.md). The architectural endpoint is one private authoritative native storage model for `f64`, `f32`, and all eight integer classes, not indefinite maintenance of the compatibility mirrors described below.

## Status vocabulary

- **Verified**: implemented with focused exact-storage or poisoned-mirror tests.
- **Implemented**: a typed path exists, but the full per-operation matrix has not been rerun in this recovery pass.
- **Rejected**: the operation deliberately returns an error instead of coercing through an `f64` compatibility buffer.
- **Open audit**: broad support exists, but exhaustive closure evidence is not yet available.

## Per-dtype matrix

The storage and dispatch implementation is shared across the eight integer classes. Tests named below use class matrices where practical and boundary tests for `i64`/`u64`.

| Surface | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense storage, shape, reshape | Verified | Verified | Verified | Verified | Verified | Verified | Verified | Verified |
| Scalar and dense casts | Verified | Verified | Verified | Verified | Verified | Verified | Verified | Verified |
| Dense indexing and assignment | Verified | Verified | Verified | Verified | Verified | Verified | Verified | Verified |
| Concatenation and shape moves | Verified | Verified | Verified | Verified | Verified | Verified | Verified | Verified |
| Arithmetic and comparison | Verified | Verified | Verified | Verified | Verified | Verified | Verified | Verified |
| Native reductions | Verified | Verified | Verified | Verified | Verified | Verified | Verified | Verified |
| Sparse storage and MAT persistence | Verified | Verified | Verified | Verified | Verified | Verified | Verified | Verified |
| Display and JSON/WASM previews | Verified | Verified | Verified | Verified | Verified | Verified | Verified | Verified |
| CPU/provider upload and download | Implemented | Implemented | Implemented | Verified | Implemented | Implemented | Implemented | Verified |
| In-process GPU shape operations | Implemented | Implemented | Implemented | Verified | Implemented | Implemented | Implemented | Verified |
| WGPU shape/reduction operations | Open audit | Open audit | Open audit | Verified | Open audit | Open audit | Open audit | Verified |
| Builtin input and option parsing | Open audit | Open audit | Open audit | Verified | Open audit | Open audit | Open audit | Verified |

## Verified compatibility areas

- `IntegerStorage` and `IntValue` cover all signed and unsigned widths.
- Dense real tensors preserve integer storage through construction, casts, indexing, assignment, reshape, concatenation, comparisons, arithmetic, and native reductions.
- Complex integer tensors preserve paired typed storage through supported structural, display, and persistence operations.
- Unsupported complex integer arithmetic is rejected before consulting the compatibility mirror.
- Sparse integer tensors preserve class and exact values through construction, transpose, dense conversion, display, MAT save/load, network writes, and WASM serialization.
- VM selectors, expression indexing, cell indexing, and `end` conversion read typed storage and reject values that cannot be represented by the relevant host index or exact-double boundary.
- Provider transfers use typed upload/download paths. Typed operations must remain resident or reject/fall back before a lossy result conversion.
- In-process provider `permute`, `flip`, `circshift`, and `repmat` preserve integer registries without allocating `f64` mirrors.
- WGPU coverage includes typed upload/download and focused wide `i64`/`u64` shape and reduction paths.
- Broad builtin parser sweeps cover numeric counts, dimensions, flags, table metadata, statistics, optimization, plotting, I/O, strings, and signal processing.

Primary conformance suites:

- `crates/runmat-runtime/tests/integer_conformance.rs`
- `crates/runmat-runtime/tests/data_integer_persistence.rs`
- integer-storage unit tests in `runmat-builtins`, `runmat-runtime`, `runmat-vm`, `runmat-accelerate`, and `runmat-wasm`

## Current representation and authoritative-storage target

The current integer implementation uses dual representations in several host value types. This is the migration starting point, not the target architecture. “Mirror” means a lossy compatibility view retained for legacy floating consumers; it never means a second source of truth.

| Layer/value | Current authority | Transitional/secondary state | Target |
| --- | --- | --- | --- |
| `Value::Int` | `IntValue` native scalar variant | None | Keep native until an explicit typed conversion |
| `Tensor` double | `data: Vec<f64>` | Separate mutable dtype/shape caches | Private `F64(Vec<f64>)`; dtype derived from storage |
| `Tensor` single | widened `data: Vec<f64>` plus `F32` metadata | No native host `f32` payload | Private `F32(Vec<f32>)`; dtype derived from storage |
| `Tensor` integer | `integer_data: Option<IntegerStorage>` | eager `data: Vec<f64>` mirror plus dtype | One private native integer storage variant; no mirror |
| `ComplexTensor` | floating pairs or paired integer storage | floating pair mirror for integers | Private typed complex storage with enforced class/length |
| `SparseTensor` | CSC plus floating or integer values | floating values mirror for integers | CSC plus one private typed values payload |
| `DataArrayValues` | typed enum variants | explicit `to_f64_vec` conversion | Preserve typed authority and convert only at declared boundaries |
| `GpuTensorHandle` | provider buffer plus external metadata registries | precision/type facts can be detached from handle | Handle/provider state directly owns authoritative numeric metadata |
| WGPU buffer | native float or packed `u32` integer words | two words for 64-bit integers | Keep backend layout; dispatch from shared numeric type contract |
| WASM/JSON preview | typed value extraction | JSON number where safe; decimal string above JS safe range | Preserve exact type/value through wire contract |

Phase 1 has introduced `NumericStorage` with exhaustive native `F64`, `F32`, and eight integer variants, plus immutable and mutable typed views, dtype/length/byte-size derivation, overflow-safe shape validation, and lossless adapters from the transitional `IntegerStorage`. `Tensor` has not yet moved onto this field; that separation keeps the type-definition slice buildable before Phase 2 compiler enumeration.

The floating mirror is legitimate only at a documented conversion to a floating result/domain. Examples include `double(A)`, a builtin whose integer input is documented to produce double, plotting geometry, or a numerical algorithm that explicitly accepts integer input by conversion. A consumer that uses a mirror for integer comparison, ordering, hashing, indexing, assignment, class-preserving arithmetic, serialization, or transfer is defective.

### Current mechanical census

Run the stable census from the repository root:

```bash
scripts/development/integer-storage-census.sh
```

The July 31 pre-migration baseline is recorded in [`integer-storage-migration-ledger.md`](./integer-storage-migration-ledger.md). It includes 667 files constructing dense tensors, a strong lower bound of 248 files with named direct-data access, 320 runtime files mentioning floating materialization, 357 files using `HostTensorView`, and 35 files using `HostIntegerTensorView`.

These patterns overlap and include tests, unrelated `data` fields, and intentional floating algorithms. They are search frontiers, not defect counts or semantic progress measures. Phase 2 field privatization and compiler diagnostics establish the authoritative direct-access inventory.

### Threading status by boundary

| Boundary | Current state | Remaining proof/work |
| --- | --- | --- |
| Host construction and casts | Exact storage is authoritative and broadly tested | Finish constructor/function-specific semantic fixes from the research ledger |
| Arithmetic, comparison, ordering | Dedicated exact helpers exist, including mixed numeric comparison tests | Complete generated scalar/1x1/array/class cross-products and remove bypass paths |
| Indexing and assignment | Selectors and typed assignment have exact paths | Audit remaining VM real/complex RHS materializers and prove they are reached only for floating destinations |
| Shape and concatenation | Broad exact support | Finish mixed representation, empty, N-D, and class-dominance conformance |
| Reductions and scans | Native exact CPU/provider paths exist | Complete all classes/options/dimensions and reject unsupported floating-only forms |
| Sparse | Exact integer CSC storage is implemented | Classify it as a RunMat extension at MATLAB-facing constructors and interchange boundaries |
| Complex integers | Exact host structural/persistence support exists | GPU upload is rejected; arithmetic remains deliberately unsupported |
| MAT and raw binary I/O | Exact Level-5 and typed binary paths are substantial | Complete MAT-version surface, complex binary layout, typed import options, and exact external-format tests |
| Data/replay/plot payloads | `DataArrayValues` is typed | Scene hydration currently calls `to_f64_vec` and discards dtype; decide and implement an explicit floating boundary or typed scene payload |
| WASM/host previews | Wide scalars are emitted as decimal strings | Audit every array/complex/sparse preview and round trip, not just scalars |
| Provider transfer | `upload_integer`/`download_integer` are explicit and exact | Ensure every fallback and derived handle propagates type metadata |
| WGPU execution | Packed exact arithmetic/shape/reduction kernels exist | Finish the complete operation-by-class matrix; avoid assuming representative 64-bit tests cover narrower classes |
| GPU fallback policy | Transparent auto-offload correctly permits CPU fallback | Explicit `gpuArray` and auto-offload share a handle with no provenance; MATLAB residency/error parity cannot yet be selected independently |
| Builtin dispatch policy | Parser modes and error namespaces exist | Builtin extension signatures do not receive a general compatibility-mode gate |
| Integer RNG | Integer outputs and exact uploads exist | WGPU and host range selection still use floating span arithmetic; implement unbiased full-width integer sampling |

### File-level hotspot map

| Path | Role | Current disposition |
| --- | --- | --- |
| `crates/runmat-builtins/src/lib.rs` | Defines `IntValue`, `IntegerStorage`, dense/sparse/complex dual storage | Authoritative model; legacy floating accessors remain deliberate audit boundaries |
| `crates/runmat-runtime/src/builtins/common/tensor.rs` | Shared tensor element/shape helpers | Critical choke point; exact-aware helpers should replace new direct field reads |
| `crates/runmat-runtime/src/builtins/logical/rel/integer_comparison.rs` | Exact integer/mixed-numeric comparisons | Guarded exact path with wide/poisoned-mirror tests; ensure all public comparison dispatch reaches it first |
| `crates/runmat-runtime/src/builtins/array/sorting_sets/integer_order.rs` | Exact ordering/set keys | Exact path; use as the model for other grouping/hash consumers |
| `crates/runmat-runtime/src/builtins/math/elementwise/integer_arithmetic.rs` | Class-preserving integer arithmetic | Exact core exists; function-specific scalar-double and division edges remain |
| `crates/runmat-runtime/src/builtins/math/reduction/integer_native.rs` | Native integer reductions | Exact CPU helper; provider and option cross-product remains |
| `crates/runmat-vm/src/indexing/*` | Selector conversion and assignment | Exact selector/destination paths exist; real/complex RHS materializers still need reachability proofs |
| `crates/runmat-runtime/src/builtins/array/grouping.rs` | Group keys, binning, `accumarray`, group outputs | Mixed: exact keys/labels exist, but generic value helpers convert storage to `f64`; classify per output contract |
| `crates/runmat-runtime/src/data/mod.rs` | Typed chunk/data-array payloads | Exact typed variants exist; conversion to requested `f64` is explicit |
| `crates/runmat-runtime/src/replay/scene.rs` | Scene data hydration | Confirmed dtype-erasing `to_f64_vec` boundary; decide whether scene geometry is intentionally floating or must remain typed |
| `crates/runmat-runtime/src/builtins/io/mat/*` | MAT parsing/writing | Exact integer payload encoding/decoding is substantial; version/backend surface remains |
| `crates/runmat-runtime/src/builtins/io/filetext/{fread,fwrite}.rs` | Raw typed binary I/O | Exact integer flatten/read paths and wide tests exist; remaining precision aliases/complex layout are separate gaps |
| `crates/runmat-runtime/src/builtins/io/tabular/readmatrix.rs` | Text/spreadsheet numeric import | Integer-like conversion exists but documented integer `OutputType` and wide lexical parsing remain incomplete |
| `crates/runmat-runtime/src/builtins/common/gpu_helpers.rs` | Host/provider transfer | Exact real integer upload path is correct; typed complex integer upload is explicitly rejected |
| `crates/runmat-accelerate-api/src/lib.rs` | Provider contract and handle registries | Exact integer contract is explicit; handle provenance is absent |
| `crates/runmat-accelerate/src/backend/wgpu/provider/ops/integer.rs` | Packed integer kernels | Exact arithmetic/comparison/reduction/scan implementation; complete matrix still required |
| `crates/runmat-accelerate/src/backend/wgpu/provider/ops/random.rs` | Device RNG | Floating range parameters and `2^53` span limit are confirmed |
| `crates/runmat-accelerate/src/simple_provider.rs` | Host reference/fallback provider | Typed registry exists; every fallback must prove it selects typed methods |
| `crates/runmat-wasm/src/wire/value.rs` | Host JSON representation | Wide scalars use decimal strings; aggregate preview/round-trip audit remains |
| `crates/runmat-parser/src/options.rs` and `runmat-config/.../language.rs` | Compatibility modes | Parser/config policy exists, but builtin option gating is not threaded |

## Storage migration closure protocol

Every compiler-ledger boundary must receive one recorded disposition:

1. **Native typed path.**
2. **Explicit numeric cast.**
3. **Floating computation domain.**
4. **Bounded structural parameter.**
5. **Rejected input.**
6. **Backend encoding.**
7. **Compatibility-gated extension.**
8. **Test-only typed inspection.**
9. **Removed or superseded.**

The migration ledger defines the evidence required for each disposition. Closure requires private storage, compiler diagnostics, a maintained ledger, and tests—not a falling grep count. New generic numeric helpers accept typed views or declare an explicit conversion/domain contract; they do not accept `Vec<f64>` without provenance.

## Recovery ledger

The interrupted parallel pass originally left 82 dirty writer worktrees. Their tracked states were captured and compared against the branch before cleanup. As of the July 31 Phase 0 baseline, only the primary `end-july-work` worktree exists.

- 166 file patches were already present or merged to a no-op.
- 70 patches overlapped later work and were reconciled by tracing their added helpers/tests and exact-storage behavior into the current tree.
- The current sparse network-write result was committed as `d0c82a92d`.
- Seven useful historical residual files were corrected, tested, and committed as `c8063adee`.
- Missing wide-integer `vecnorm` order handling and bounded scatterplot axes parsing were committed as `fa031e80f`.
- WASM real/sparse integer lengths were recovered during the mirror-read audit.

Archived changes deliberately not integrated:

- a visibility-only `integer_word_count` change with no remaining caller;
- a comment-only Toeplitz change;
- a complex integer `ldivide` test that contradicted the deliberate rejection policy;
- a graph test that treated `u64::MAX` as unrepresentable on 64-bit hosts;
- an optimization test that rejected an exact integer value that fits `usize`.

## Remaining closure work

The eval-loop epic must remain active. This branch is the furthest recovered integer lineage, not proof of exhaustive MATLAB parity.

- Complete a generated operation-by-dtype conformance run for every matrix cell rather than relying on shared implementations and representative class tests.
- Finish the WGPU/provider matrix for every operation that can preserve residency, including parity, no-download, error, and resource-cleanup cases.
- Repeat the builtin mirror-read audit after later `dev`/`main` reconciliation; those changes are intentionally outside this branch recovery.
- Replace dense, complex, and sparse compatibility mirrors with the private authoritative storage model, using compiler diagnostics to populate and close the migration ledger.
- Add a durable lint or generated inventory that flags new direct mirror reads in exact semantic domains.
- Decide how explicit `gpuArray` intent is represented separately from auto-offload residency before claiming MATLAB unsupported-call parity.
- Thread compatibility mode into builtin signature/option dispatch before relying on mode-gated integer extensions.
- Replace floating-span integer RNG generation before treating full-width `int64`/`uint64` `randi` as a completed extension.
- Decide product scope for complex integer arithmetic beyond structural and persistence support. Until then, deterministic rejection is the supported behavior.
- Reconcile `end-july-work` with the later `dev`/`main` histories only after this branch passes its standalone verification gates.
