---
title: "Integer dtype compatibility status"
category: "Development"
section: "14.6"
last_updated: "July 29, 2026"
---

# Integer dtype compatibility status

This document is the durable status and recovery ledger for the first-class
integer work on `end-july-work`. The work tracks eval-loop issue
`ri-b4e5076574b9`.

The branch forks from the integer commit that was subsequently merged through
`dev` and `main` (`66ead7582`). It intentionally does not include later `dev`
or `main` changes.

## Status vocabulary

- **Verified**: implemented with focused exact-storage or poisoned-mirror tests.
- **Implemented**: a typed path exists, but the full per-operation matrix has
  not been rerun in this recovery pass.
- **Rejected**: the operation deliberately returns an error instead of
  coercing through an `f64` compatibility buffer.
- **Open audit**: broad support exists, but exhaustive closure evidence is not
  yet available.

## Per-dtype matrix

The storage and dispatch implementation is shared across the eight integer
classes. Tests named below use class matrices where practical and boundary
tests for `i64`/`u64`.

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
- Dense real tensors preserve integer storage through construction, casts,
  indexing, assignment, reshape, concatenation, comparisons, arithmetic, and
  native reductions.
- Complex integer tensors preserve paired typed storage through supported
  structural, display, and persistence operations.
- Unsupported complex integer arithmetic is rejected before consulting the
  compatibility mirror.
- Sparse integer tensors preserve class and exact values through construction,
  transpose, dense conversion, display, MAT save/load, network writes, and
  WASM serialization.
- VM selectors, expression indexing, cell indexing, and `end` conversion read
  typed storage and reject values that cannot be represented by the relevant
  host index or exact-double boundary.
- Provider transfers use typed upload/download paths. Typed operations must
  remain resident or reject/fall back before a lossy result conversion.
- In-process provider `permute`, `flip`, `circshift`, and `repmat` preserve
  integer registries without allocating `f64` mirrors.
- WGPU coverage includes typed upload/download and focused wide `i64`/`u64`
  shape and reduction paths.
- Broad builtin parser sweeps cover numeric counts, dimensions, flags, table
  metadata, statistics, optimization, plotting, I/O, strings, and signal
  processing.

Primary conformance suites:

- `crates/runmat-runtime/tests/integer_conformance.rs`
- `crates/runmat-runtime/tests/data_integer_persistence.rs`
- integer-storage unit tests in `runmat-builtins`, `runmat-runtime`,
  `runmat-vm`, `runmat-accelerate`, and `runmat-wasm`

## Recovery ledger

The interrupted parallel pass left 82 dirty writer worktrees. Their tracked
states were captured and compared against the branch before cleanup.

- 166 file patches were already present or merged to a no-op.
- 70 patches overlapped later work and were reconciled by tracing their added
  helpers/tests and exact-storage behavior into the current tree.
- The current sparse network-write result was committed as `d0c82a92d`.
- Seven useful historical residual files were corrected, tested, and committed
  as `c8063adee`.
- Missing wide-integer `vecnorm` order handling and bounded scatterplot axes
  parsing were committed as `fa031e80f`.
- WASM real/sparse integer lengths were recovered during the mirror-read audit.

Archived changes deliberately not integrated:

- a visibility-only `integer_word_count` change with no remaining caller;
- a comment-only Toeplitz change;
- a complex integer `ldivide` test that contradicted the deliberate rejection
  policy;
- a graph test that treated `u64::MAX` as unrepresentable on 64-bit hosts;
- an optimization test that rejected an exact integer value that fits `usize`.

## Remaining closure work

The eval-loop epic must remain active. This branch is the furthest recovered
integer lineage, not proof of exhaustive MATLAB parity.

- Complete a generated operation-by-dtype conformance run for every matrix
  cell rather than relying on shared implementations and representative class
  tests.
- Finish the WGPU/provider matrix for every operation that can preserve
  residency, including parity, no-download, error, and resource-cleanup cases.
- Repeat the builtin mirror-read audit after later `dev`/`main` reconciliation;
  those changes are intentionally outside this branch recovery.
- Decide product scope for complex integer arithmetic beyond structural and
  persistence support. Until then, deterministic rejection is the supported
  behavior.
- Reconcile `end-july-work` with the later `dev`/`main` histories only after
  this branch passes its standalone verification gates.
