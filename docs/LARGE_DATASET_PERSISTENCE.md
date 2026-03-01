# Large Dataset Persistence

This document defines the cloud-native persistence model for very large numeric datasets in RunMat.

The design targets:

- local-to-cloud portability without rewriting user code,
- selective read/write for multi-GB to multi-PB arrays,
- strong integrity and reproducibility guarantees,
- collaborative workflows where many users and jobs operate on shared data,
- tight integration with plotting/replay so 3D and large scenes remain practical.

## Why this exists

RunMat currently supports filesystem-backed `save`/`load`, artifact persistence, and project snapshots. That covers many workflows but leaves two gaps:

1. Very large arrays are awkward as monolithic files (costly rewrites, slow random reads).
2. Figure replay payloads need a scalable data substrate for large 3D/stateful visuals.

We need a first-class dataset layer, not only file IO conveniences.

## Design goals

- **Scalable by default**: efficient random reads/writes for TB/PB datasets.
- **Portable semantics**: same user code works on local and remote backends.
- **Low rewrite cost**: mutate subsets without rewriting full files.
- **Deterministic history**: snapshots and versions produce reproducible states.
- **Collaboration-ready**: support concurrent usage and future real-time editing.
- **Performance-aware**: parallel chunk reads/writes, compression, caching.

## Non-goals (initial phases)

- Perfect compatibility with every MATLAB `.mat` variant as the primary format.
- Full distributed query engine in v1.
- Arbitrary schema evolution without explicit versioning rules.
- CRDT collaboration for all data operations on day one.

## Guiding principles

1. **Data is not just a file**: represent datasets as manifests plus chunk objects.
2. **Immutable heavy blobs, mutable lightweight metadata**.
3. **Explicit API for large data** (avoid overloading `save/load` semantics).
4. **Content-addressed objects** for dedupe, integrity, and reproducibility.
5. **Transaction boundaries are explicit** and observable.

## Terminology

- **Dataset**: logical container at a path like `/datasets/weather.data`.
- **Array**: named N-dimensional tensor inside a dataset.
- **Chunk**: fixed-shape subregion of an array stored as one object.
- **Manifest**: metadata document defining arrays, chunk grid, codec, and references.
- **Snapshot**: project-level immutable checkpoint already provided by RunMat Server.

## Format: RunMat Data (`.data`)

`*.data` is a logical dataset root represented by structured metadata plus chunk objects.

Recommended structure:

```
/datasets/weather.data/
  manifest.json
  arrays/
    temperature/
      meta.json
      chunks/
        0.0.0
        0.0.1
        ...
    pressure/
      meta.json
      chunks/
  attrs/
    tags.json
    provenance.json
```

At remote scale, chunk payloads may map to object-store keys and not require materializing a directory tree client-side.

## Data model

### Dataset manifest

`manifest.json` includes:

- `schema_version`
- dataset id and optional title/description
- array registry (`name -> array meta reference`)
- creation/update timestamps
- optional transaction id / parent manifest id
- optional collaboration metadata (future)

### Array metadata

Per-array metadata includes:

- `dtype` (f32/f64/i64/u8/bool/...)
- `shape`
- `chunk_shape`
- memory order (`column_major` default to match MATLAB expectations)
- codec (`none`, `lz4`, `zstd`, future)
- fill value
- optional statistics/min-max for planning

### Chunk payloads

Chunks are stored as immutable objects referenced by hash/object id.

Chunk descriptor fields:

- chunk key/index (e.g. `2.8.0`)
- object id/hash
- uncompressed bytes
- compressed bytes
- checksum

## API surface in RunMat language

We should keep `save/load` for convenience and compatibility, but add explicit large-data APIs.

### User-facing module

```
ds = data.open("/datasets/weather.data");
T = ds.array("temperature");

block = T(1:1024, :, 10);
T(1:1024, :, 10) = new_block;

tx = ds.begin();
tx.write("temperature", {1:1024, :, 10}, new_block);
tx.write_attr("note", "calibration v2");
tx.commit();
```

### Suggested functions

- `data.create(path, schema)`
- `data.open(path)`
- `data.exists(path)`
- `data.copy(from, to, options)`
- `data.snapshot(path, label)` (dataset-level convenience over project snapshots)
- `data.export(path, "mat")` / `data.import(path, "mat")` for interoperability

### `save/load` policy

- Keep `save/load` unchanged for small/local workflows and compatibility.
- Add guidance: for large persistent tensors, prefer `data.*` APIs.
- Optionally allow `save(..., "format", "data")` as a bridge path later.

## Read/write semantics

### Reads

- Lazy and slice-oriented.
- Planner maps slice to chunk set.
- Parallel chunk fetch with bounded concurrency.
- Optional client cache keyed by object hash.

### Writes

- Region writes update only touched chunks.
- Two-phase flow:
  1. upload new/changed chunk objects,
  2. atomically commit new manifest pointer.
- Failed commits leave no partially visible dataset state.

### Consistency model

- Default: snapshot-isolation-like semantics at manifest boundaries.
- Readers observe a stable manifest view.
- Writers commit via compare-and-swap on manifest version.

## Performance and scale strategy

### Chunk sizing

- configurable per array, with sensible defaults by dtype and dimensionality.
- target balancing random access and throughput (not one-size-fits-all).

### Compression

- default `zstd` for cold storage and bandwidth efficiency.
- optional `lz4` for low-latency interactive reads.
- no compression for already compressed domains.

### Transport

- Reuse server upload-session APIs for parallel direct chunk upload.
- Reuse download-url/ranged read paths for high-throughput reads.
- Avoid API-server bottlenecks for very large transfers.

### Caching

- local object cache keyed by chunk hash.
- optional read-ahead for sequential scans.
- explicit cache controls for memory-constrained clients.

## Collaboration model

For "Figma for math" style collaboration, separate two planes:

1. **Data plane (heavy, immutable chunks)**
2. **Intent/annotation plane (lightweight, mutable ops)**

### Data collaboration

- Concurrent writers use manifest CAS and transaction conflicts are explicit.
- Merge strategies can be array-region aware in future phases.

### View/state collaboration (future)

- camera, overlays, labels, plot styling, selections, bookmarks as ops.
- operation log can be streamed over existing event infrastructure.
- periodic checkpoints compact op history.

This keeps giant tensor payloads stable while enabling real-time shared editing semantics.

## Integration with plotting and replay

Large/3D figure scenes should reference dataset-backed buffers instead of embedding large numeric payloads inline.

Benefits:

- small scene descriptors,
- efficient reuse across runs,
- replay/import can stream only needed geometry slices,
- clear fallback path (`preview-only`) when unsupported scene kinds are encountered.

## Security and governance

- enforce project ACLs on dataset paths and object references.
- short-lived upload/download URLs.
- checksums required on chunk uploads.
- audit logs for manifest commits and snapshot/restore events.
- retention policy must account for object references from snapshots.

## Observability

Add explicit telemetry for:

- chunk read/write throughput and latency,
- cache hit/miss,
- manifest commit conflicts,
- bytes uploaded/downloaded per operation,
- scan amplification (requested slice vs transferred bytes),
- per-array hot spots.

## Compatibility and ecosystem

- Provide import/export bridges for `.mat` and common scientific formats.
- Keep on-disk/on-wire schema versioned and backward readable.
- publish a stable machine-readable schema for tooling.

## Rollout plan

### Phase 0: RFC + schema

- finalize manifest and chunk schema,
- decide baseline codec and chunk index conventions,
- define transaction and conflict semantics.

### Phase 1: read path

- open dataset, inspect metadata, slice reads,
- object cache and basic telemetry,
- CLI inspection tools.

### Phase 2: write path

- region writes and atomic manifest commit,
- upload-session integration,
- retry/idempotency handling.

### Phase 3: history + snapshots

- dataset-aware history UX,
- fast restore and branching semantics,
- retention and GC safety checks.

### Phase 4: plotting integration

- figure scenes referencing dataset buffers,
- full 3D replay support,
- explicit replay capability metadata.

### Phase 5: collaboration ops

- shared annotation/view state protocol,
- conflict handling policy,
- checkpoint/compaction strategy.

## Key design tradeoffs

- **Inline simplicity vs referenced buffers**: references are more complex but scale far better.
- **Strict transactions vs eventual writes**: strict commits reduce ambiguity but require conflict UX.
- **Aggressive compression vs low latency**: choose per workload/profile.
- **MATLAB familiarity vs explicit APIs**: explicit `data.*` is clearer for large-scale semantics.

## Decisions

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

## Success criteria

- Read/write subregions of multi-terabyte arrays without full-file rewrite.
- Deterministic reproducibility through manifest versioning and snapshots.
- Efficient remote operation using direct chunk transfer and caching.
- Plot/replay pipeline can represent large 3D data without blank rehydration outcomes.
- User-facing API is understandable, explicit, and productive for both small and huge workloads.

## RFC appendix: schema

This appendix defines concrete wire/on-disk shapes used for implementation planning.

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

### Optional collaboration op-log entry

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

## RFC appendix: API contract

### Data module

- `data.create(path, schema, options?) -> Dataset`
- `data.open(path, options?) -> Dataset`
- `data.exists(path) -> logical`
- `data.copy(fromPath, toPath, options?) -> logical`
- `data.import(path, format, sourcePath, options?) -> Dataset`
- `data.export(path, format, targetPath, options?) -> logical`

### Dataset object

- `ds.arrays() -> string[]`
- `ds.array(name) -> DataArray`
- `ds.attrs() -> map`
- `ds.setAttr(key, value) -> logical`
- `ds.begin() -> DataTransaction`
- `ds.version() -> string`

### Array object

- `A.shape() -> double[]`
- `A.dtype() -> string`
- `A.read(sliceSpec, options?) -> matrix`
- `A.write(sliceSpec, values, options?) -> logical`

### Transaction object

- `tx.write(arrayName, sliceSpec, values) -> logical`
- `tx.setAttr(key, value) -> logical`
- `tx.commit(options?) -> logical`
- `tx.abort() -> logical`

## RFC appendix: error model

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

Suggested error payload:

```json
{
  "code": "MANIFEST_CONFLICT",
  "message": "manifest changed since transaction start",
  "retryable": true,
  "details": {
    "path": "/datasets/weather.data",
    "expectedManifest": "sha256:aaa...",
    "actualManifest": "sha256:bbb..."
  },
  "traceId": "4f87..."
}
```

## RFC appendix: migration strategy

### From `save/load` to `data.*`

- keep `save/load` default behavior unchanged,
- add docs + lints/hints for large arrays,
- provide `data.import(..., "mat", ...)` and `data.export(..., "mat", ...)` for interop,
- optional future bridge: `save(..., "format", "data")`.

### Existing artifacts and replay

- continue supporting current scene artifact format,
- progressively introduce scene references to dataset chunk objects,
- mark replay capability explicitly (`full`, `preview-only`) to avoid blank states.

## Full user-facing API surface

This section is the implementation contract for runtime/CLI/desktop UX and language tooling.

### Module functions

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

### Dataset object

- `ds.path() -> string`
- `ds.id() -> string`
- `ds.version() -> string` (manifest hash/version id)
- `ds.arrays() -> string[]`
- `ds.has_array(name) -> logical`
- `ds.array(name) -> DataArray`
- `ds.attrs() -> map`
- `ds.get_attr(key, default?) -> any`
- `ds.set_attr(key, value) -> logical`
- `ds.set_attrs(map) -> logical`
- `ds.begin(options?) -> DataTransaction`
- `ds.snapshot(label?, options?) -> string` (returns snapshot/version id)
- `ds.refresh() -> Dataset<T>`

### Array object

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
- indexing sugar (equivalent to read/write):
  - `x = A(i, j, k, ...)`
  - `A(i, j, k, ...) = values`

### Transaction object

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

### Options and behavior guarantees

- `read` options: `consistency`, `cache`, `prefetch`, `timeout_ms`
- `write` options: `atomic`, `idempotency_key`, `timeout_ms`
- `commit` options: `if_manifest`, `retry_policy`
- default write safety is atomic commit at manifest boundary.

## Typing model and rules

### Compile-time schema inference

- If `data.open("literal/path.data")` has compile-time constant path and schema is available, typechecker resolves the schema and returns `Dataset<ResolvedSchema>`.
- If path is dynamic, return `Dataset<dynamic>` unless caller supplies explicit schema.
- If path is constant but schema cannot be resolved in strict mode, emit a type error.

### Explicit schema overrides

- `data.open(path, schema)` forces typed handle and enables static checks.
- runtime validates supplied schema against manifest and errors on incompatible mismatch.

### Type propagation

- `ds.array("temperature")` resolves to `DataArray<f32, [N, M, K]>` when schema known.
- `A.read(slice)` returns element/container type derived from dtype and resulting rank.
- scalar index reduces rank; range/slice preserves rank.
- writes require assignable dtype and shape-compatible RHS.

### Shape typing

- exact dimensions when literals are known.
- symbolic dimensions when unknown at compile-time.
- linter enforces high-confidence mismatch detection even with symbolic dims.

### Attr typing

- dataset attrs may be typed by schema (`attrsSchema`) or dynamic map.
- in typed mode, `set_attr`/`get_attr` are validated against declared attr types.

### Transaction typing

- `tx.write` must validate dtype/shape against declared array schema.
- creating/resizing arrays in tx updates inferred schema only after successful commit.

## Lint rules

These lints are part of the default data profile.

### Type/contract lints

- `data/no-untyped-open`
  - warn/error on `data.open(path)` when schema cannot be statically resolved.
- `data/unknown-array-name`
  - invalid array key for typed datasets.
- `data/implicit-lossy-cast`
  - writing with narrowing or precision-loss cast.
- `data/shape-mismatch`
  - slice assignment dimensions incompatible with target.
- `data/invalid-slice-rank`
  - rank/index mismatch for array operations.

### Safety/consistency lints

- `data/ignore-commit-result`
  - commit outcome must be checked or handled.
- `data/no-multiwrite-outside-tx`
  - multiple related writes without explicit transaction.
- `data/no-stale-manifest-assumption`
  - warns when code assumes current manifest without refresh/check.

### Performance lints

- `data/no-full-read-large-array`
  - full materialization of large arrays without explicit opt-in.
- `data/no-tiny-slice-loop`
  - repeated small random slice reads in tight loops.
- `data/missing-prefetch-hint`
  - sequential large scan without prefetch/read strategy.

### Collaboration lints

- `data/mutable-view-without-op`
  - direct mutation of collaboration state outside op API.
- `data/conflict-policy-implicit`
  - write path with unspecified conflict handling in shared contexts.

## Error handling contract (user-facing)

- Recoverable conflicts:
  - `MANIFEST_CONFLICT`, `TXN_ABORTED`, `QUOTA_EXCEEDED` (context-dependent)
- Input/type errors:
  - `INVALID_SLICE`, `DTYPE_MISMATCH`, `SHAPE_MISMATCH`, `ARRAY_NOT_FOUND`
- Integrity and storage errors:
  - `CHECKSUM_MISMATCH`, `CHUNK_MISSING`, `PERMISSION_DENIED`

User code should be able to pattern-match by `err.code` reliably across runtime and server-backed execution.

## Compatibility policy for `save/load`

- `save/load` remain available and stable.
- lint suggestions recommend `data.*` for large arrays and collaborative datasets.
- interop helpers (`data.import/export`) are the standard bridge for MAT workflows.

## Ownership and abstraction boundaries

This section defines where logic lives across the RunMat codebase so implementation remains maintainable and consistent.

### Compiler and language layer

- Owns the user-facing language contract for `data.*`.
- Owns static typing for `Dataset<T>`, `DataArray<T, Shape>`, and transaction call typing.
- Owns compile-time schema inference for constant-path `data.open("...")`.
- Owns lint rule evaluation and diagnostics.

Expected location:

- parser/HIR/type/lint crates in `runmat/crates/*`.

### Runtime execution layer

- Owns runtime semantics for dataset operations (`open`, `read`, `write`, `tx.commit`, CAS conflict behavior).
- Owns slice planning, chunk selection, commit state machine, and in-process caching.
- Owns normalized error mapping for data operations.

Expected location:

- `runmat/crates/runmat-runtime` under a dedicated `data` subsystem.

### Filesystem/provider layer

- Owns byte transport and storage operations only.
- Does not own dataset semantics, typing, or transaction policy.
- Exposes stable primitives used by runtime data subsystem.

Expected location:

- existing filesystem providers and remote provider abstractions.

### Plotting and replay layer

- Owns figure scene serialization/import contracts, including 3D support.
- Owns capability metadata (`full` vs `preview-only`) and dataset-buffer references in scenes.
- Does not own dataset transaction semantics.

Expected location:

- `runmat/crates/runmat-plot` and related replay runtime modules.

### Client UX layer (desktop/CLI/browser)

- Owns user workflows, discovery, progress, and error presentation.
- Consumes runtime/server behavior; does not implement divergent dataset semantics.
- Surfaces lint/type diagnostics and conflict resolution UX.

Expected location:

- `runmat-private/desktop`, CLI UX surfaces, browser host components.

## Boundary rules

- Compiler must not depend on provider-specific transport behavior.
- Runtime must not encode UI-specific state transitions.
- Server must not encode language typechecking policy.
- Client UX must not fork core dataset semantics.
- Error codes and transaction semantics must be consistent across runtime, CLI, desktop, and server-backed execution.

## Suggested implementation module map

### Runtime data subsystem

- `runtime/data/schema` (schema structs, validation, compatibility checks)
- `runtime/data/array` (typed array handles and read/write entry points)
- `runtime/data/slice` (slice normalization and planning)
- `runtime/data/chunk_index` (chunk lookup/index management)
- `runtime/data/txn` (transaction state machine + CAS commit)
- `runtime/data/errors` (canonical error enum and mapping)
- `runtime/data/interop_mat` (MAT import/export bridge)

### Compiler and lints

- `semantic/data_signatures` (`data.*` function contracts)
- `semantic/data_schema_infer` (constant-path schema resolver)
- `types/data_types` (`Dataset<T>`, `DataArray<T, Shape>`)
- `lints/data/*` (rule implementations and diagnostics)

## Minimal examples (final API)

```matlab
schema = data.schema(struct(
  "arrays", struct(
    "temperature", struct("dtype", "f32", "shape", [4096, 4096, 365], "chunk", [256, 256, 1]),
    "pressure", struct("dtype", "f32", "shape", [4096, 4096, 365], "chunk", [256, 256, 1])
  ),
  "attrs", struct("owner", "string", "stage", "string")
));

ds = data.create("/datasets/weather.data", schema);
A = ds.array("temperature");
A(1:256, :, 1) = rand(256, 4096);

tx = ds.begin();
tx.write("pressure", {1:256, :, 1}, rand(256, 4096));
tx.set_attr("stage", "calibrated");
ok = tx.commit();
```

```matlab
% Dynamic path, explicit schema to keep type safety
path = strcat("/datasets/", run_id, ".data");
ds = data.open(path, schema);
```

## Finalized implementation decisions

These are the highest-impact choices that are now fixed for implementation:

1. **Adopt `.data` as first-class dataset container** with explicit `data.*` API.
2. **Use content-addressed chunk objects + manifest CAS commits** as core write model.
3. **Default array order to `column_major`** for user expectation alignment.
4. **Use `zstd` default codec** with per-array override (`lz4` for interactive hot paths).
5. **Treat `save/load` as compatibility path**, not the primary large-data API.
6. **Separate immutable data plane and mutable collaboration ops plane**.
7. **Integrate plotting replay via dataset references** rather than large inline scene payloads.

## Implementation checklist

- Keep API names clear and MATLAB-idiomatic while preserving explicit semantics.
- Document transaction/conflict behavior in user-facing runtime docs and errors.
- Support implicit single-write commits and explicit transactions for multi-write units.
- Implement dataset snapshots as wrappers over project snapshot primitives.
- Encode default chunking heuristics in one planner module with deterministic behavior.
- Ensure error codes are returned consistently across runtime, CLI, desktop, and server APIs.

## Current implementation status

- Runtime now persists array payloads with chunk sidecars (`arrays/<name>/chunks/index.json`) in addition to manifest metadata, and reconstructs reads from chunk indexes when present.
- Filesystem provider abstraction now exposes provider-neutral data transport primitives (`data_manifest_descriptor`, `data_chunk_upload_targets`, `data_upload_chunk`) with concrete implementations across native/sandbox/remote providers.
- Runtime `data.*` API now includes `data.copy`, `data.move`, `data.import`, `data.export`, plus transaction operations for `resize`, `fill`, `create_array`, and `delete_array`.
- Lint coverage includes `data/no-multiwrite-outside-tx` in addition to untyped-open and commit guidance lints.
- Lint coverage now also includes manifest-informed checks for unknown array names (`data/unknown-array-name`) and invalid slice rank (`data/invalid-slice-rank`) when `data.open('<literal-path>')` can resolve a local manifest.
- Runtime test coverage includes HTTP endpoint integration for touched-chunk uploads, including cross-boundary slice writes that upload only intersecting chunk keys.
- Chunk hashes now use SHA-256 (`sha256:<hex>`) to match server contract expectations.
- Runtime emits structured tracing events (`target=runmat.data`) for transaction begin/commit/abort, manifest conflicts, chunk planning, and chunk upload completions.
- Domain-specific static analysis has been extracted from `runmat-hir` into `runmat-static-analysis` with modular files (`schema`, `lints/data_api`, `lints/shape`) to keep HIR crate boundaries clean.

## Done matrix

- API surface (`data.*`, `Dataset`, `DataArray`, `DataTransaction`): **done**
- Provider abstraction + concrete providers (native/sandbox/remote/wasm fallback): **done**
- Server `/data/manifest` + `/data/chunks/upload-targets`: **done**
- N-D chunk grid planner + touched-chunk-only slice writes: **done**
- Manifest CAS / conflict semantics (`txn_sequence`, `if_manifest`): **done**
- SHA-256 hash parity on chunk descriptors: **done**
- Typed resolver/inference coverage for data methods/objects with `Dataset<T>` / `DataArray<T,Shape>` semantics: **done**
- Manifest-informed lints (`unknown-array-name`, `invalid-slice-rank`, multiwrite): **done**
- Runtime HTTP integration tests for touched chunk uploads: **done**
- Observability hooks for core data operations: **done (tracing events)**

## Final validation checklist

Run these from `runmat/` unless noted:

1. `cargo fmt`
2. `cargo test -p runmat-runtime --lib builtins::io::data:: -- --nocapture`
3. `cargo test -p runmat-runtime --lib data:: -- --nocapture`
4. `cargo test -p runmat-hir --test type_inference -- --nocapture`
5. `cargo test -p runmat-filesystem --lib -- --nocapture`
6. `cargo test -p runmat-lsp diagnostics_include_shape_lints -- --nocapture`
7. From `runmat-private/server/`: `cargo test -p server-http --test filesystem -- --nocapture`
