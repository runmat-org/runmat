# Large Dataset Persistence: Internal Reference

This file contains internal implementation reference for the large dataset persistence system. For user-facing documentation, see [LARGE_DATASET_PERSISTENCE.md](LARGE_DATASET_PERSISTENCE.md).

---

## Design rationale

RunMat supports filesystem-backed `save`/`load`, artifact persistence, and project snapshots. That covers many workflows but leaves two gaps:

1. Very large arrays are awkward as monolithic files (costly rewrites, slow random reads).
2. Figure replay payloads need a scalable data substrate for large 3D/stateful visuals.

The `data.*` API and `.data` format address both gaps with a chunked, content-addressed persistence model.

### Design goals

- Scalable by default: efficient random reads/writes for TB/PB datasets.
- Portable semantics: same user code works on local and remote backends.
- Low rewrite cost: mutate subsets without rewriting full files.
- Deterministic history: snapshots and versions produce reproducible states.
- Collaboration-ready: support concurrent usage and future real-time editing.
- Performance-aware: parallel chunk reads/writes, compression, caching.

### Non-goals (initial phases)

- Perfect compatibility with every MATLAB `.mat` variant as the primary format.
- Full distributed query engine in v1.
- Arbitrary schema evolution without explicit versioning rules.
- CRDT collaboration for all data operations on day one.

### Guiding principles

1. Represent datasets as manifests plus chunk objects, not monolithic files.
2. Heavy blobs are immutable; lightweight metadata is mutable.
3. Large data gets an explicit API (`data.*`) rather than overloading `save/load` semantics.
4. Content-addressed objects provide deduplication, integrity, and reproducibility.
5. Transaction boundaries are explicit and observable.

---

## Key design tradeoffs

- Inline simplicity vs referenced buffers: references are more complex but scale far better for large payloads.
- Strict transactions vs eventual writes: strict CAS commits reduce ambiguity but require conflict UX.
- Aggressive compression vs low latency: `zstd` for cold storage, `lz4` for interactive hot paths.
- MATLAB familiarity vs explicit APIs: explicit `data.*` is clearer for large-scale semantics than overloading `save/load`.

---

## Finalized implementation decisions

1. `.data` is a first-class logical dataset path with virtual internals. Hosts may materialize a directory view, but the API contract is logical-path based.
2. Default chunking is dimensionality-aware and write-safe:
   - 1D arrays target 8-32 MB chunk bytes,
   - 2D arrays default to square-ish chunks targeting 8-16 MB,
   - 3D+ arrays default to slab chunks favoring the leading two dimensions and shallow depth,
   - callers can override at create time per array.
3. Default codec is `zstd` (level 3) with per-array override; `lz4` is recommended for interactive hot paths.
4. `save/load` remain compatibility APIs. Large-scale persistence is explicitly `data.*`, with optional import/export bridges.
5. Concurrent writes use optimistic manifest CAS. Overlapping writes fail with `MANIFEST_CONFLICT` by default; explicit merge policy can be added later.
6. Dataset snapshots are first-class API calls implemented as thin wrappers over project snapshot primitives.
7. Plot replay uses dataset references for large payloads and capability metadata (`full` vs `preview-only`) to avoid blank rehydration states.

---

## Success criteria

- Read/write subregions of multi-terabyte arrays without full-file rewrite.
- Deterministic reproducibility through manifest versioning and snapshots.
- Efficient remote operation using direct chunk transfer and caching.
- Plot/replay pipeline can represent large 3D data without blank rehydration outcomes.
- User-facing API is understandable, explicit, and productive for both small and huge workloads.

---

## Rollout plan

### Phase 0: RFC + schema

- Finalize manifest and chunk schema.
- Decide baseline codec and chunk index conventions.
- Define transaction and conflict semantics.

### Phase 1: read path

- Open dataset, inspect metadata, slice reads.
- Object cache and basic telemetry.
- CLI inspection tools.

### Phase 2: write path

- Region writes and atomic manifest commit.
- Upload-session integration.
- Retry/idempotency handling.

### Phase 3: history + snapshots

- Dataset-aware history UX.
- Fast restore and branching semantics.
- Retention and GC safety checks.

### Phase 4: plotting integration

- Figure scenes referencing dataset buffers.
- Full 3D replay support.
- Explicit replay capability metadata.

### Phase 5: collaboration ops

- Shared annotation/view state protocol.
- Conflict handling policy.
- Checkpoint/compaction strategy.

---

## Observability

Structured tracing events are emitted under `target=runmat.data` for:

- chunk read/write throughput and latency,
- cache hit/miss rates,
- manifest commit conflicts,
- bytes uploaded/downloaded per operation,
- scan amplification (requested slice vs transferred bytes),
- per-array hot spots,
- transaction begin/commit/abort lifecycle.

---

## Ownership and abstraction boundaries

This section defines where logic lives across the codebase so implementation remains maintainable and consistent.

### Compiler and language layer

- Owns the user-facing language contract for `data.*`.
- Owns static typing for `Dataset<T>`, `DataArray<T, Shape>`, and transaction call typing.
- Owns compile-time schema inference for constant-path `data.open("...")`.
- Owns lint rule evaluation and diagnostics.

Expected location: parser/HIR/type/lint crates in `runmat/crates/*`.

### Runtime execution layer

- Owns runtime semantics for dataset operations (`open`, `read`, `write`, `tx.commit`, CAS conflict behavior).
- Owns slice planning, chunk selection, commit state machine, and in-process caching.
- Owns normalized error mapping for data operations.

Expected location: `runmat/crates/runmat-runtime` under a dedicated `data` subsystem.

### Filesystem/provider layer

- Owns byte transport and storage operations only.
- Does not own dataset semantics, typing, or transaction policy.
- Exposes stable primitives used by runtime data subsystem.

Expected location: existing filesystem providers and remote provider abstractions.

### Plotting and replay layer

- Owns figure scene serialization/import contracts, including 3D support.
- Owns capability metadata (`full` vs `preview-only`) and dataset-buffer references in scenes.
- Does not own dataset transaction semantics.

Expected location: `runmat/crates/runmat-plot` and related replay runtime modules.

### Client UX layer (desktop/CLI/browser)

- Owns user workflows, discovery, progress, and error presentation.
- Consumes runtime/server behavior; does not implement divergent dataset semantics.
- Surfaces lint/type diagnostics and conflict resolution UX.

Expected location: `runmat-private/desktop`, CLI UX surfaces, browser host components.

---

## Boundary rules

- Compiler must not depend on provider-specific transport behavior.
- Runtime must not encode UI-specific state transitions.
- Server must not encode language typechecking policy.
- Client UX must not fork core dataset semantics.
- Error codes and transaction semantics must be consistent across runtime, CLI, desktop, and server-backed execution.

---

## Suggested implementation module map

### Runtime data subsystem

- `runtime/data/schema` -- schema structs, validation, compatibility checks
- `runtime/data/array` -- typed array handles and read/write entry points
- `runtime/data/slice` -- slice normalization and planning
- `runtime/data/chunk_index` -- chunk lookup/index management
- `runtime/data/txn` -- transaction state machine + CAS commit
- `runtime/data/errors` -- canonical error enum and mapping
- `runtime/data/interop_mat` -- MAT import/export bridge

### Compiler and lints

- `semantic/data_signatures` -- `data.*` function contracts
- `semantic/data_schema_infer` -- constant-path schema resolver
- `types/data_types` -- `Dataset<T>`, `DataArray<T, Shape>`
- `lints/data/*` -- rule implementations and diagnostics

---

## Implementation checklist

- Keep API names clear and MATLAB-idiomatic while preserving explicit semantics.
- Document transaction/conflict behavior in user-facing runtime docs and errors.
- Support implicit single-write commits and explicit transactions for multi-write units.
- Implement dataset snapshots as wrappers over project snapshot primitives.
- Encode default chunking heuristics in one planner module with deterministic behavior.
- Ensure error codes are returned consistently across runtime, CLI, desktop, and server APIs.

---

## Current implementation status

- Runtime persists array payloads with chunk sidecars (`arrays/<name>/chunks/index.json`) in addition to manifest metadata, and reconstructs reads from chunk indexes when present.
- Filesystem provider abstraction exposes provider-neutral data transport primitives (`data_manifest_descriptor`, `data_chunk_upload_targets`, `data_upload_chunk`) with concrete implementations across native/sandbox/remote providers.
- Runtime `data.*` API includes `data.copy`, `data.move`, `data.import`, `data.export`, plus transaction operations for `resize`, `fill`, `create_array`, and `delete_array`.
- Lint coverage includes `data/no-multiwrite-outside-tx` in addition to untyped-open and commit guidance lints.
- Lint coverage includes manifest-informed checks for unknown array names (`data/unknown-array-name`) and invalid slice rank (`data/invalid-slice-rank`) when `data.open('<literal-path>')` can resolve a local manifest.
- Runtime test coverage includes HTTP endpoint integration for touched-chunk uploads, including cross-boundary slice writes that upload only intersecting chunk keys.
- Chunk hashes use SHA-256 (`sha256:<hex>`) to match server contract expectations.
- Runtime emits structured tracing events (`target=runmat.data`) for transaction begin/commit/abort, manifest conflicts, chunk planning, and chunk upload completions.
- Domain-specific static analysis is extracted from `runmat-hir` into `runmat-static-analysis` with modular files (`schema`, `lints/data_api`, `lints/shape`) to keep HIR crate boundaries clean.

---

## Done matrix

| Area | Status |
|---|---|
| API surface (`data.*`, `Dataset`, `DataArray`, `DataTransaction`) | done |
| Provider abstraction + concrete providers (native/sandbox/remote/wasm fallback) | done |
| Server `/data/manifest` + `/data/chunks/upload-targets` | done |
| N-D chunk grid planner + touched-chunk-only slice writes | done |
| Manifest CAS / conflict semantics (`txn_sequence`, `if_manifest`) | done |
| SHA-256 hash parity on chunk descriptors | done |
| Typed resolver/inference for data methods with `Dataset<T>` / `DataArray<T,Shape>` | done |
| Manifest-informed lints (`unknown-array-name`, `invalid-slice-rank`, multiwrite) | done |
| Runtime HTTP integration tests for touched chunk uploads | done |
| Observability hooks for core data operations | done (tracing events) |

---

## Final validation checklist

Run these from `runmat/` unless noted:

1. `cargo fmt`
2. `cargo test -p runmat-runtime --lib builtins::io::data:: -- --nocapture`
3. `cargo test -p runmat-runtime --lib data:: -- --nocapture`
4. `cargo test -p runmat-hir --test type_inference -- --nocapture`
5. `cargo test -p runmat-filesystem --lib -- --nocapture`
6. `cargo test -p runmat-lsp diagnostics_include_shape_lints -- --nocapture`
7. From `runmat-private/server/`: `cargo test -p server-http --test filesystem -- --nocapture`

---

## RFC appendix: schema reference

These are the concrete wire/on-disk shapes used across the implementation.

### Dataset manifest (`manifest.json`)

```json
{
  "schemaVersion": 1,
  "format": "runmat-data",
  "datasetId": "ds_01HR8F6R8Q7Q6J7W9J8Y",
  "name": "weather-sim",
  "createdAt": "2026-03-01T00:00:00Z",
  "updatedAt": "2026-03-01T00:00:00Z",
  "parentManifest": null,
  "manifestHash": "sha256:ad4...",
  "arrays": {
    "temperature": {
      "metaPath": "arrays/temperature/meta.json",
      "latestChunkSet": "sha256:4fd..."
    },
    "pressure": {
      "metaPath": "arrays/pressure/meta.json",
      "latestChunkSet": "sha256:be1..."
    }
  },
  "attrs": {
    "path": "attrs/tags.json"
  },
  "txn": {
    "lastCommitId": "tx_01HR8F6V5CG",
    "sequence": 42
  },
  "capabilities": {
    "sparse": false,
    "collabOps": false
  }
}
```

### Array metadata (`arrays/<name>/meta.json`)

```json
{
  "schemaVersion": 1,
  "name": "temperature",
  "dtype": "f32",
  "shape": [4096, 4096, 365],
  "chunkShape": [256, 256, 1],
  "order": "column_major",
  "codec": {
    "name": "zstd",
    "level": 3
  },
  "fillValue": 0.0,
  "stats": {
    "min": -48.2,
    "max": 52.9
  },
  "chunkIndex": {
    "path": "arrays/temperature/chunks/index.json",
    "hash": "sha256:3de..."
  }
}
```

### Chunk index (`arrays/<name>/chunks/index.json`)

```json
{
  "schemaVersion": 1,
  "array": "temperature",
  "chunks": [
    {
      "key": "0.0.0",
      "shape": [256, 256, 1],
      "dtype": "f32",
      "encoding": "zstd",
      "objectId": "obj_2f3f...",
      "hash": "sha256:1aa...",
      "bytesRaw": 262144,
      "bytesStored": 49321,
      "checksum": "sha256:1aa..."
    }
  ]
}
```

### Collaboration op-log entry

```json
{
  "schemaVersion": 1,
  "opId": "op_01HR8F7B0W",
  "ts": "2026-03-01T00:01:20Z",
  "actorId": "usr_123",
  "datasetId": "ds_01HR8F6R8Q7Q6J7W9J8Y",
  "type": "set-attr",
  "path": "/views/main/camera",
  "value": {
    "azimuth": 35,
    "elevation": 20,
    "distance": 8.5
  },
  "baseManifest": "sha256:ad4..."
}
```

### Full API contract (expanded)

#### Module functions

- `data.create(path, schema, options?) -> Dataset<Schema>`
- `data.open(path, schemaOrOptions?) -> Dataset<T>`
- `data.exists(path) -> logical`
- `data.delete(path, options?) -> logical`
- `data.copy(fromPath, toPath, options?) -> logical`
- `data.move(fromPath, toPath, options?) -> logical`
- `data.import(path, format, sourcePath, options?) -> Dataset<dynamic>`
- `data.export(path, format, targetPath, options?) -> logical`
- `data.list(pathPrefix?, options?) -> string[]`
- `data.inspect(path) -> struct`

#### Dataset object

- `ds.path() -> string`
- `ds.id() -> string`
- `ds.version() -> string`
- `ds.arrays() -> string[]`
- `ds.has_array(name) -> logical`
- `ds.array(name) -> DataArray`
- `ds.attrs() -> map`
- `ds.get_attr(key, default?) -> any`
- `ds.set_attr(key, value) -> logical`
- `ds.set_attrs(map) -> logical`
- `ds.begin(options?) -> DataTransaction`
- `ds.snapshot(label?, options?) -> string`
- `ds.refresh() -> Dataset<T>`

#### Array object

- `A.name() -> string`
- `A.dtype() -> string`
- `A.shape() -> double[]`
- `A.rank() -> double`
- `A.chunk_shape() -> double[]`
- `A.codec() -> string`
- `A.read(sliceSpec, options?) -> matrix`
- `A.write(sliceSpec, values, options?) -> logical`
- `A.resize(newShape, options?) -> logical`
- `A.fill(value, sliceSpec?, options?) -> logical`
- Indexing sugar: `x = A(i, j, k, ...)` / `A(i, j, k, ...) = values`

#### Transaction object

- `tx.id() -> string`
- `tx.write(arrayName, sliceSpec, values, options?) -> logical`
- `tx.resize(arrayName, newShape, options?) -> logical`
- `tx.fill(arrayName, value, sliceSpec?, options?) -> logical`
- `tx.set_attr(key, value) -> logical`
- `tx.set_attrs(map) -> logical`
- `tx.delete_array(name) -> logical`
- `tx.create_array(name, meta) -> logical`
- `tx.commit(options?) -> logical`
- `tx.abort() -> logical`
- `tx.status() -> string` (`open`, `committed`, `aborted`)

#### Options and behavior guarantees

- `read` options: `consistency`, `cache`, `prefetch`, `timeout_ms`
- `write` options: `atomic`, `idempotency_key`, `timeout_ms`
- `commit` options: `if_manifest`, `retry_policy`
- Default write safety is atomic commit at manifest boundary.

### Typing model

#### Compile-time schema inference

- If `data.open("literal/path.data")` has compile-time constant path and schema is available, typechecker resolves the schema and returns `Dataset<ResolvedSchema>`.
- If path is dynamic, return `Dataset<dynamic>` unless caller supplies explicit schema.
- If path is constant but schema cannot be resolved in strict mode, emit a type error.

#### Type propagation

- `ds.array("temperature")` resolves to `DataArray<f32, [N, M, K]>` when schema known.
- `A.read(slice)` returns element/container type derived from dtype and resulting rank.
- Scalar index reduces rank; range/slice preserves rank.
- Writes require assignable dtype and shape-compatible RHS.

#### Shape typing

- Exact dimensions when literals are known.
- Symbolic dimensions when unknown at compile-time.
- Linter enforces high-confidence mismatch detection even with symbolic dims.

#### Attr typing

- Dataset attrs may be typed by schema (`attrsSchema`) or dynamic map.
- In typed mode, `set_attr`/`get_attr` are validated against declared attr types.

#### Transaction typing

- `tx.write` must validate dtype/shape against declared array schema.
- Creating/resizing arrays in tx updates inferred schema only after successful commit.

### Lint rules (full specification)

#### Type/contract lints

- `data/no-untyped-open` -- warn/error on `data.open(path)` when schema cannot be statically resolved.
- `data/unknown-array-name` -- invalid array key for typed datasets.
- `data/implicit-lossy-cast` -- writing with narrowing or precision-loss cast.
- `data/shape-mismatch` -- slice assignment dimensions incompatible with target.
- `data/invalid-slice-rank` -- rank/index mismatch for array operations.

#### Safety/consistency lints

- `data/ignore-commit-result` -- commit outcome must be checked or handled.
- `data/no-multiwrite-outside-tx` -- multiple related writes without explicit transaction.
- `data/no-stale-manifest-assumption` -- warns when code assumes current manifest without refresh/check.

#### Performance lints

- `data/no-full-read-large-array` -- full materialization of large arrays without explicit opt-in.
- `data/no-tiny-slice-loop` -- repeated small random slice reads in tight loops.
- `data/missing-prefetch-hint` -- sequential large scan without prefetch/read strategy.

#### Collaboration lints

- `data/mutable-view-without-op` -- direct mutation of collaboration state outside op API.
- `data/conflict-policy-implicit` -- write path with unspecified conflict handling in shared contexts.

### Error model (full specification)

Standardized error categories for CLI/runtime/API layers:

- `DATASET_NOT_FOUND`
- `ARRAY_NOT_FOUND`
- `INVALID_SLICE`
- `DTYPE_MISMATCH`
- `SHAPE_MISMATCH`
- `CHUNK_MISSING`
- `CHECKSUM_MISMATCH`
- `MANIFEST_CONFLICT`
- `TXN_ABORTED`
- `PERMISSION_DENIED`
- `QUOTA_EXCEEDED`
- `UNSUPPORTED_FEATURE`
- `INTERNAL_ERROR`
