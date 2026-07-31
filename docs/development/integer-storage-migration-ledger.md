---
title: "Integer Storage Migration Ledger"
category: "Development"
section: "14.7"
last_updated: "July 31, 2026"
---

# Integer storage migration ledger

This is the in-repository ledger contract for replacing numeric compatibility mirrors with one authoritative native storage model. It complements [`integer-dtype-status.md`](./integer-dtype-status.md): the status document describes architecture and semantic coverage, while this ledger records the disposition of concrete migration boundaries.

The external operator plan and current dashboard live outside the RunMat repository in `../MATLAB_INTEGER_PLAN.md` and `../MATLAB_INTEGER_PROGRESS.md`.

## Baseline

- Branch: `end-july-work`
- Pre-migration HEAD: `c574e41b817cc58d729b96505008ce1958ad88bf`
- Baseline date: July 31, 2026
- Census command: `scripts/development/integer-storage-census.sh`
- Exact compiler inventory: pending Phase 2 storage privatization

The lexical census is a discovery frontier. It contains tests, unrelated `data` fields, intentional floating algorithms, and repeated uses within one file. Its counts must not be subtracted from one another and reported as semantic completion.

## Phase 1 storage foundation

`runmat-builtins` now defines the migration target's real numeric primitives:

- `NumericStorage` has native `F64`, `F32`, `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`, and `U64` variants.
- `NumericStorageView` and `NumericStorageViewMut` expose matching typed slices without an implicit floating conversion.
- Dtype, class name, element count, and byte count derive from the storage variant.
- Shape validation checks both element count and `usize` multiplication overflow.
- Moving between the transitional `IntegerStorage` and integer `NumericStorage` variants preserves the original vectors and wide values without an `f64` intermediate.

The current `Tensor` fields remain unchanged in this foundation slice. Phase 2 will place storage behind the private authoritative field and populate the compiler ledger.

## Entry schema

The Phase 2 compiler inventory will populate entries using this schema:

| Field | Required content |
| --- | --- |
| ID | Stable `ISL-NNNN` identifier |
| Layer | Builtins, VM, compiler, runtime, serialization, accelerate API, provider, WGPU, WASM, or other explicit owner |
| Path and symbol | Repository-relative file plus containing function/type/test |
| Access kind | Construction, immutable read, mutable read, replacement, dtype, shape, transfer, persistence, display, or test inspection |
| Numeric classes | `F64`, `F32`, integer class set, complex, sparse, or all numeric |
| Semantic IDs | Applicable resolution/TODO identifiers, or `architecture-only` |
| Disposition | One of the closed dispositions below |
| Verification | Focused test plus shared check proving the disposition |
| Commit | Slice commit that closed the entry |
| Status | Pending, in progress, closed, or superseded |
| Notes | Bounded rationale, including why floating conversion is valid when used |

Entries are grouped by migration boundary, not mechanically one row per compiler diagnostic. Several diagnostics may close together only when they share one typed API, semantic rule, and verification.

## Closed dispositions

Every compiler-ledger entry must end in exactly one of these states:

1. **Native typed path** — reads or mutates the authoritative storage variant without conversion.
2. **Explicit numeric cast** — implements a documented MATLAB class conversion and tests value/class behavior, including wide integer or `f32` rounding sentinels where applicable.
3. **Floating computation domain** — the public operation accepts the input but computes/returns in a documented floating domain; conversion is explicit at that boundary.
4. **Bounded structural parameter** — exact input is range-checked before conversion to `usize`, `u32`, an exact `f64`, or another bounded host type.
5. **Rejected input** — validation rejects the class before an incompatible storage accessor runs.
6. **Backend encoding** — provider/WGPU code converts between authoritative typed values and an exact backend representation such as packed `u32` words.
7. **Compatibility-gated extension** — behavior is intentionally outside MATLAB compatibility, documented, gated, and tested.
8. **Test-only typed inspection** — a test inspects a specific storage variant without creating a production escape hatch.
9. **Removed or superseded** — dead or duplicate access disappears, with the replacement entry/commit named.

“Legacy `f64` accessor” and “kept for convenience” are not closed dispositions.

## Inventory procedure

1. Run the lexical census and record its output in the current progress entry.
2. Introduce the private authoritative storage API in a buildable slice.
3. Privatize legacy storage/dtype fields at the selected crate boundary.
4. Capture machine-readable compiler diagnostics.
5. Normalize diagnostics into stable boundary entries using the schema above.
6. Close entries through typed APIs and semantic tests.
7. Re-run compiler diagnostics until the selected boundary is clean.
8. Re-run the lexical census only as a regression/search aid.
9. At each phase boundary, confirm that all entries introduced in that phase are closed or explicitly carried forward with an owner and reason.

## Phase 0 guardrail evidence

Existing tests already exercise authoritative integer storage with cleared or poisoned floating mirrors across:

- all eight dense integer classes in `runmat-builtins`;
- dense and complex VM indexing and assignment;
- selectors, `end`, cell indexing, and structural graph metadata;
- runtime comparison, ordering, statistics, construction, and I/O helpers;
- auto-promotion and exact provider download; and
- packed wide-integer WGPU structural operations.

Phase 0 adds a native-`single` semantic sentinel without freezing the transitional `Vec<f64>` physical representation. Phase 1 must preserve those values and dtype while moving the payload to native `Vec<f32>` storage.

### Verification baseline

The following checks passed on July 31, 2026:

```text
bash -n scripts/development/integer-storage-census.sh
scripts/development/integer-storage-census.sh
cargo fmt --all -- --check
cargo test -p runmat-builtins integer_storage_tests
cargo test -p runmat-runtime --test integer_conformance
cargo test -p runmat-runtime --test data_integer_persistence
cargo test -p runmat-vm poisoned
cargo test -p runmat-accelerate mirrorless
cargo test -p runmat-accelerate poisoned_f64_mirror
cargo check --workspace
git diff --check
```

No pre-existing failure was found in these Phase 0 gates. This is not a claim that the full workspace test matrix is green; later phases retain their own focused and workspace-wide exit criteria.

## Initial frontier

Run `scripts/development/integer-storage-census.sh` for the authoritative current output. At the pre-migration baseline it reports:

| Metric | Files | Matching lines | Matches |
| --- | ---: | ---: | ---: |
| Dense tensor constructors | 667 | 6,858 | 6,858 |
| Strong named direct data | 248 | 1,338 | 1,438 |
| Runtime `Value::Tensor` | 672 | 9,059 | 9,532 |
| Runtime `integer_storage(...)` | 332 | 979 | 994 |
| Runtime floating materialization | 320 | 1,014 | 1,031 |
| Floating provider view | 357 | 1,563 | 1,563 |
| Integer provider view | 35 | 168 | 168 |

These rows use different patterns and overlap. They are not additive.
