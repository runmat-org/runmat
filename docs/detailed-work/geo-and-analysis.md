# Geometry Loading and Analysis Architecture

## Purpose

This document defines a long-lived architecture for geometry-native workflows in RunMat,
including geometry loading, structured geometry operations, and future FEA/physics analysis.

## Goals

- Treat geometry as a first-class project resource, not a viewer-only artifact.
- Provide a stable operation surface shared by UI/runtime/harness layers.
- Keep importer/parser logic reusable and independent from UI.
- Support scalable compute execution (CPU + GPU) with clear residency semantics.
- Enable deterministic, reproducible analysis workflows with artifact lineage.
- Lower the barrier to advanced simulation by making geometry + analysis intent-driven and interactive.
- Push simulation throughput with GPU-first execution paths where numerically appropriate.

## Strategic Ambition

RunMat aims to make high-end numerical simulation substantially more accessible than legacy specialist tools,
while growing toward comparable or better technical depth over time.

This implies two simultaneous priorities:

- **Breadth and rigor**: eventual support for broad analysis classes and strong numerical trust.
- **Usability and speed**: integrated workflows that collapse setup/solve/visualize loops and maximize GPU residency.

The architecture in this document is designed to support both without locking into monolithic solver design.

## Non-goals (for initial phases)

- Full multiphysics coverage in first release.
- Perfectly lossless import/export across every CAD/mesh format from day one.
- UI-level one-off logic that bypasses runtime operation contracts.

## Conceptual Model

### Core terms

- **Geometry Asset**: canonical project-stored geometry representation.
- **Source Geometry**: original CAD/mesh semantics as imported (may include assembly + material hints).
- **Derived Geometry**: tessellated/render/analysis-ready forms derived from source geometry.
- **Representation**: view/interaction mode for a resource (text, image, model3d, etc.).
- **Inspector**: identifies resource kind and confidence (mime/ext/signature/metadata).
- **Handler**: operation provider for a representation.
- **Capability**: explicit supported actions (`view`, `edit`, `inspect`, `snapshot`, `analyze`).
- **Analysis Model**: solver-agnostic description of material, BC, load, and step intent.
- **Analysis Run**: concrete solve execution with backend, diagnostics, and results.
- **Tessellation Profile**: deterministic settings used to convert CAD/B-rep into mesh forms.
- **Material Evidence**: provenance+confidence wrapper for imported material assignments/properties.

### Layering rule

- Format-specific behavior lives at the edges (`geometry-io` adapters).
- Canonical data/ops live in middle layers (`geometry-core`, `geometry-ops`, `analysis-core`).
- Consumers (UI/runtime/tools/server) use stable operation contracts.

## Workspace Layout Convention

To keep `runmat/crates` organized as domains expand, crate directories should be grouped by domain namespace:

- `crates/runmat-geometry/core`
- `crates/runmat-geometry/io`
- `crates/runmat-geometry/ops`
- `crates/runmat-analysis/core`
- `crates/runmat-analysis/fea`

Package names remain explicit (`runmat-geometry-core`, `runmat-analysis-fea`, etc.) for dependency clarity.

Future domains should follow the same pattern (e.g. `runmat-meshing/*`, `runmat-multiphysics/*`).

## Crate Architecture

### `runmat-geometry-core`

Canonical domain model for geometry and field data.

Proposed modules:

- `model/geometry.rs` (`GeometryAsset`, `GeometryId`, source metadata)
- `model/mesh.rs` (`SurfaceMesh`, `VolumeMesh`, connectivity)
- `model/topology.rs` (`Vertex`, `Edge`, `Face`, `Cell`, element kinds)
- `model/regions.rs` (`Region`, `RegionId`, tags/material groups)
- `model/attributes.rs` (typed per-entity attributes)
- `model/field.rs` (`Field`, location semantics: node/element/face)
- `model/xform.rs` (`Transform`, local/world frames)
- `model/units.rs` (unit descriptors and conversion helpers)
- `selection/` (`EntityRef`, query spec primitives)
- `diagnostics/` (`Diagnostic`, severity/code conventions)

Key properties:

- no parser IO,
- no UI dependencies,
- no solver assumptions,
- deterministic serialization capability.

### `runmat-geometry-io`

Format import/export + normalization.

Proposed modules:

- `sniff.rs` (content signature + mime/ext resolution)
- `import/stl.rs`
- `import/obj.rs`
- `import/ply.rs`
- `import/gltf.rs` (phase-gated)
- `normalize.rs` (vertex dedupe, winding/normals corrections)
- `report.rs` (`ImportReport`, warnings, recoverable errors)

Output contract:

- `GeometryAsset`
- `ImportReport`

### `runmat-geometry-ops`

Pure geometry operations.

Proposed modules:

- `bounds.rs` (AABB/OBB helpers)
- `stats.rs` (counts, memory footprints, complexity)
- `quality.rs` (degenerate/non-manifold/quality checks)
- `spatial.rs` (acceleration structures, proximity queries)
- `queries.rs` (selection and region filters)
- `prep.rs` (analysis-oriented prep helpers)

### `runmat-analysis-core`

Solver-agnostic analysis/problem model.

Proposed modules:

- `problem/model.rs` (`AnalysisModel`, `AnalysisModelId`)
- `problem/materials.rs` (`MaterialModel`, assignment semantics)
- `problem/bc.rs` (boundary conditions)
- `problem/loads.rs` (force, pressure, body loads, thermal placeholders)
- `problem/steps.rs` (static/modal/transient/nonlinear step definitions)
- `schema/results.rs` (expected result field definitions)
- `validate.rs` (completeness and consistency checks)

### `runmat-analysis-fea`

Concrete FEA engines + assembly/solve/post.

Proposed modules:

- `discretize/` (mesh-to-element mapping)
- `assembly/` (global system assembly)
- `solve/linear.rs` (direct/iterative)
- `solve/nonlinear.rs` (incremental + convergence)
- `post/fields.rs` (stress/strain/displacement recovery)
- `post/derived.rs` (von Mises, principal values)
- `diagnostics/` (residuals, condition warnings)

## Long-Term Expansion Domains (Planned)

To approach broad simulation parity with incumbent tools, the following domains are required beyond initial FEA:

### `runmat-meshing-*`

- surface/volume mesh generation,
- adaptivity and refinement,
- quality optimization,
- boundary-layer generation for flow/thermal classes.

### `runmat-multiphysics-*`

- coupling orchestration across physics domains,
- timestep/iteration coupling strategies,
- transfer operators between discretizations.

### `runmat-cad-interop-*`

- CAD/B-rep ingestion bridges,
- geometry healing/defeaturing,
- robust topology mapping from CAD to analysis geometry.

End-state objective:

- support seamless loading of common engineering-native CAD ecosystems through pluggable import adapters,
- preserve assembly hierarchy, units, and part metadata,
- preserve material assignments and available property metadata with confidence levels,
- maintain performant import/tessellation paths with deterministic provenance.

Required abstractions:

- source-vs-derived geometry separation,
- assembly tree contract (`AssemblyNode`),
- tessellation profile contract (`TessellationProfile`),
- material ingestion contract with provenance (`MaterialEvidence`).

### `runmat-hpc-runtime-*`

- distributed solve orchestration,
- checkpoint/restart,
- domain decomposition and multi-node scheduling.

### `runmat-benchmarks-*`

- verification fixture corpus,
- parity/tolerance governance,
- reproducibility and regression gating.

## Runtime Integration (`runmat-runtime`)

Runtime orchestrates resource operations and analysis runs while preserving reproducibility.

Responsibilities:

- resource open/inspect routing,
- geometry load lifecycle,
- analysis model/run lifecycle,
- artifact/provenance capture,
- backend/device selection,
- cancellation and retries,
- telemetry/event emission.

## Unified Operation Surface

These operations should be treated as stable public contracts used by:

- UI viewers,
- runtime user logic,
- tool integrations,
- server-hosted execution.

Initial operation set:

- `resource.inspect(path)`
- `geometry.load(path)`
- `geometry.inspect(handle)`
- `geometry.list_regions(handle)`
- `geometry.query_entities(handle, query)`
- `geometry.compute_stats(handle)`
- `geometry.capture_view(handle, view_spec)`
- `analysis.create_model(geometry_handle, intent_spec)`
- `analysis.validate(model_handle)`
- `analysis.run(model_handle, run_spec)`
- `analysis.results(run_id, field_query)`

Solver implementation direction (v1.1 progression):

- adopt operator-centric internals (`apply_k`, `apply_m`, `apply_c`) as the stable numerical boundary
- prefer matrix-free iterative solve paths for large-scale GPU portability
- keep operation envelopes unchanged while solver internals advance

### Usage Modes (No Ambiguity)

The same contracts are intended to be consumed in four equivalent ways:

1. Desktop UI flow (interactive):
   - open file -> `geometry.inspect` -> `geometry.load`
   - author/adjust model -> `analysis.validate`
   - run solve -> `analysis.run_linear_static`
   - inspect outputs -> `analysis.results`
2. Scripted runtime flow (batch/non-UI):
   - call the same operations from runtime/harness code with explicit `trace_id` and `request_id`.
3. Tool integration flow (agent/automation):
   - compose operation calls as explicit steps; consume typed envelopes and typed errors only.
4. Remote execution flow (server):
   - submit operation requests over API; preserve `op_version`, `error_code`, and trace fields end-to-end.

No consumer may reinterpret geometry or solver semantics locally; all semantics come from operation contracts.

### Minimal v1 User Flow

1. Import a part/assembly (`geometry.load`) from STL first, STEP next.
2. Build an analysis model from geometry regions/material evidence.
3. Validate model (`analysis.validate`) against units/frame and required constructs.
4. Run linear static solve (`analysis.run_linear_static`) with explicit backend preference.
5. Consume fields/diagnostics and gate publishability on validity + convergence + quality tiers.

### Runtime Envelope Contract (v1)

Each operation returns either:

- success envelope: `{ operation, op_version, trace_id, request_id, data }`
- error envelope: `{ error_code, error_type, message, operation, op_version, retryable, severity, context, trace_id, request_id, timestamp }`

`error_code` is stable API. `message` is human-readable only.

For analysis solve responses, `data` must include:

- tier gates: `model_validity`, `solver_convergence`, `result_quality`
- publication decision: `run_status` and `publishable`
- provenance core: `backend`, `solver_backend`, `solver_device_apply_k_ratio`, `solver_host_sync_count`, `precision_mode`, `deterministic_mode`, `solver_method`, `preconditioner`, `fallback_events`

Material assignment confidence diagnostics policy:

- model may carry `material_assignments` with confidence tier (`verified`, `probable`, `inferred`)
- mismatched assignment diagnostics must be emitted with stable codes:
  - `ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_VERIFIED` (error)
  - `ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_PROBABLE` (warning)
  - `ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_INFERRED` (warning)
- any emitted assignment conflict warning/error downgrades result quality to at least `warn`

For result retrieval (`analysis.results/v1`), query/response semantics are:

- query: `include_fields` (empty = all), `include_diagnostics` (bool)
- response: `fields`, optional `diagnostics`, `run_status`, `publishable`, `provenance`, `summary`
- invalid field requests must return typed error `ANALYSIS_RESULTS_FIELD_NOT_FOUND`
- run lookup supports `run_id` retrieval and must return `ANALYSIS_RESULTS_RUN_NOT_FOUND` when lineage record is absent

Run artifact persistence boundary:

- runtime analysis writes run envelopes to an artifact store via adapter interface (`AnalysisArtifactStore`)
- default adapter is in-memory for local runtime/test execution
- filesystem adapter is available for persisted run lineage; object-store adapters can be added without contract changes
- large field payload storage remains behind adapter boundaries so storage implementation can reuse existing tensor/object pipelines

Solve requests should carry explicit run options, at minimum:

- `deterministic_mode` (true/false)
- `precision_mode` (`fp32` | `fp64` | `mixed`)
- `preconditioner_mode` (`auto` | `jacobi` | `amg` | `ilu`), with explicit fallback events when unsupported

Field payloads must use a domain field container (not raw numeric arrays in operation contracts):

- `AnalysisField { field_id, shape, values }`
- `values` supports `host_f64` and `device_ref` variants
- runtime adapters may project to host tensors or provider handles, but contract shape remains stable

GPU-targeted analysis runs should attempt to promote host-backed fields to `device_ref` values when an acceleration provider is available; when promotion is not possible, the run remains valid but must record explicit fallback events.

## Contract Versioning and Compatibility

- Every operation request/response must carry a contract version (`op_version`).
- Backward-compatible additions are additive-only (new optional fields).
- Breaking changes require a new operation version and explicit migration notes.
- Persisted artifacts must record operation version used to generate them.

## Pre-Implementation Decisions (Must Lock)

Implementation should not begin until the following are explicitly locked and reviewed:

1. Entity identity lifecycle and remap policy across geometry revisions.
2. Unit and coordinate-frame conventions (including tensor conventions).
3. v1 analysis feature matrix (accepted vs rejected model constructs).
4. Typed operation error taxonomy and retryability semantics.
5. Performance SLOs and redline thresholds.
6. Contract conformance CI between operation producers/consumers.
7. Benchmark governance ownership and approval workflow.

## Identity, Topology, and Mapping Contracts

### Entity identity invariants

- `EntityRef` values are stable only within a specific geometry revision.
- Any remesh/retopology step creates a new geometry revision and invalidates prior direct refs.
- If remapping exists, it must be explicit (`EntityRefMap`) with confidence/coverage metrics.

### Required `EntityRef` shape

`EntityRef` must include:

- `geometry_id`
- `geometry_revision`
- `mesh_id`
- `entity_kind`
- `entity_id`

References without revision context are invalid by contract.

### Geometry vs analysis mesh

- Imported geometry and analysis discretization are separate concepts.
- Result fields must always declare source mesh (`geometry_mesh` or `analysis_mesh`).
- When projected for display, projection method must be recorded in field metadata.

## Units and Coordinate Frame Policy

- Geometry, materials, loads, and outputs must each declare units.
- Runtime validation must reject incomplete or ambiguous unit sets.
- Frame semantics (global vs local region frame) must be explicit for BC/load definitions.
- Unit conversions must be deterministic and logged in provenance.

## Validation and Trust Model

Treat trust as tiered statuses, not a single boolean.

Required status dimensions:

- `import_validity` (format parse/normalization quality)
- `model_validity` (analysis setup completeness/consistency)
- `solver_convergence` (residual behavior, iteration status)
- `result_quality` (projection quality, mesh quality impact)

A run is considered publishable only if all required tiers meet configured thresholds.

### v1 acceptance matrix (analysis)

- Supported in v1:
  - linear static structural solve,
  - isotropic linear elastic material,
  - displacement BC,
  - force/pressure loads.
- Rejected in v1 (hard error):
  - nonlinear material models,
  - contact definitions,
  - transient/modal steps,
  - multiphysics couplings.

Unsupported constructs must fail validation with typed errors (no silent downgrade).

## Error Model (Strongly Typed and Traceable)

### Error envelope

Every operation error must return a typed envelope with trace linkage:

```json
{
  "error_code": "GEOMETRY_INVALID_TOPOLOGY",
  "error_type": "validation",
  "message": "Non-manifold edge detected",
  "operation": "geometry.load",
  "op_version": "geometry.load/v1",
  "retryable": false,
  "severity": "error",
  "context": {
    "geometry_id": "geo_...",
    "geometry_revision": 3,
    "entity_ref": null
  },
  "trace_id": "...",
  "request_id": "...",
  "timestamp": "..."
}
```

### Error dimensions (required)

- `error_code` (stable machine enum)
- `error_type` (`input`, `validation`, `capacity`, `backend`, `internal`)
- `retryable` (explicit)
- `severity` (`warning`, `error`, `fatal`)
- trace fields (`trace_id`, `request_id`)

### Canonical code families

- `GEOMETRY_*`
  - import/format/topology/identity issues
- `ANALYSIS_*`
  - model validation and step definition issues
- `SOLVER_*`
  - convergence/numerics/backend solve failures
- `BACKEND_*`
  - acceleration provider/device/runtime failures
- `CAPACITY_*`
  - size/memory/timeout limits
- `CONTRACT_*`
  - op version/schema compatibility errors

### Traceability rules

- Every error must be emitted as structured telemetry with same `error_code`.
- Persisted run summaries must include any terminal error envelope.
- UI/harness/server must pass through `error_code` unchanged.

## Determinism and Backend Provenance

- Provide explicit execution modes: `fast` and `reproducible`.
- `reproducible` mode enforces deterministic ordering and backend constraints.
- Every run records:
  - backend (`cpu`, `cuda`, etc.)
  - device id
  - precision mode (`fp32`, `fp64`, mixed)
  - deterministic mode flag
  - fallback events (if any)

### CPU/GPU fallback policy

- Any fallback changes run status to `degraded` unless operation is explicitly fallback-safe.
- Fallback reason must include typed `BACKEND_*` code and stage identifier.
- Fallback visibility is mandatory in result summaries.

## Failure Modes and Mitigations

### High-risk failure modes

- Identity drift after remesh leads to BC/load misattachment.
- Unit mismatch yields plausible visuals with incorrect physics.
- Non-converged solves interpreted as valid due to weak status gating.
- Importers accept malformed geometry without raising quality diagnostics.
- Silent CPU fallback causes unexpected runtime/precision differences.

### Required mitigations

- Versioned geometry revisions and explicit ref remap policies.
- Strict unit/frame validation pre-solve.
- Hard convergence/result-quality gates in result publication path.
- Mandatory import diagnostics with severity thresholds.
- Explicit backend fallback telemetry surfaced in run summaries.

## Consumer Integration Notes

Downstream consumers (UI layers, orchestration harnesses, deployment targets) should:

- call operation contracts directly,
- avoid forking geometry semantics,
- keep deterministic behavior where feasible,
- avoid UI-coupled assumptions in tool execution paths.

## GPU / Acceleration Residency

### Residency model

Define explicit residency states:

- `HostOnly`
- `DeviceResident { backend, device_id }`
- `Hybrid { host_dirty, device_dirty }`

### Policy

- promote geometry buffers once per analysis run,
- keep assembly/solve/post on device where backend supports it,
- avoid host transfer unless explicitly needed,
- fallback per-op to CPU if kernel unsupported,
- emit telemetry for promotion/eviction/fallback.

### Acceleration provider integration

Use existing acceleration provider abstractions for:

- buffer alloc/upload/download,
- kernel dispatch/streams,
- sparse/linear solve hooks,
- synchronization and memory budgeting.

### Multi-device

- choose device per run,
- keep immutable geometry shared where possible,
- isolate mutable solve state per run.

## Data Provenance and Reproducibility

For each analysis run, persist:

- geometry source hash + importer version,
- normalization flags,
- analysis model hash,
- solver/backend versions,
- runtime config (device/backend/precision),
- result field manifest.

This enables deterministic re-run comparisons and trustable lineage.

## Performance Guidance

- avoid reparsing geometry on every viewer interaction,
- build/load acceleration structures lazily with caching,
- stream large geometry ingestion,
- include early complexity guards (entity count, memory estimate),
- surface meaningful diagnostics to users before expensive runs.

## Capacity and Operational Guardrails

- Enforce model-size limits by deployment profile (desktop/server).
- Estimate memory footprint pre-run and reject unsafe executions early.
- Define timeout and cancellation semantics for every long-running operation.
- Include circuit breakers for repeated failing runs under same config.

### Initial SLO targets

- `geometry.inspect` p95 < 250 ms for <= 250 MB assets.
- `geometry.load` first-interactive p95 < 2.5 s for <= 250 MB assets.
- `analysis.validate` p95 < 400 ms for v1 model sizes.
- Solve startup overhead (excluding solver runtime) p95 < 1.0 s.

Additional CAD import SLO targets (initial):

- `geometry.load` (STEP <= 250 MB) first-interactive p95 < 5.0 s on desktop profile.
- Assembly metadata extraction p95 < 1.5 s for <= 10k-part trees.

SLO breaches should emit structured `CAPACITY_*`/`BACKEND_*` diagnostics.

## Competitive Readiness Gaps (Explicit)

Current architecture is foundational, not complete parity.

Remaining major capability gaps to track explicitly:

- multiphysics coupling depth,
- industrial meshing/adaptivity depth,
- CAD/B-rep interoperability,
- broad constitutive model library,
- distributed HPC solve maturity,
- large verification/validation benchmark coverage.

CAD-specific readiness gap:

- robust, production-grade CAD-native ingestion beyond STEP with consistent metadata fidelity and performance.

These are expected roadmap domains, not design flaws in current layering.

## Security and Safety

- treat geometry files as untrusted inputs,
- enforce parser bounds checks,
- cap maximum topology sizes per environment,
- isolate expensive operations with cancellation/timeouts,
- sanitize any embedded metadata before logging/export.

## Implementation Phases

### Phase 1: Resource groundwork

- land resource inspector/registry/viewer routing in consuming UI,
- add image viewer + binary fallback,
- stop text-decoding unknown binary.

### Phase 2: Geometry core/IO MVP

- introduce `runmat-geometry-core` + `runmat-geometry-io` (STL first),
- expose runtime geometry inspect/load ops,
- desktop `model3d` viewer reads runtime geometry handles.

### Phase 3: Geometry ops + richer formats

- add `geometry-ops` stats/queries/quality,
- add OBJ/PLY, then GLTF/GLB,
- improve scene interactions and capture operations.

### Phase 4: Analysis foundation

- introduce `runmat-analysis-core` model/validation,
- add `runmat-analysis-fea` linear static path,
- produce first-class result fields bound to geometry entities.

### Phase 5: GPU-first analysis

- promote solve path to device-resident execution where possible,
- refine fallback and telemetry,
- optimize large-model throughput.

### Phase 6: Reliability hardening

- add deterministic replay tests for representative geometry + analysis fixtures,
- add identity remap coverage checks,
- add unit/frame invariant regression tests,
- add backend parity tests (cpu vs gpu) with accepted tolerance envelopes.

### Phase 7: Expansion tracks

- land `runmat-meshing` base crates,
- land `runmat-multiphysics` coupling skeleton,
- land first distributed runtime primitives,
- formalize benchmark governance for cross-domain parity.

## Resolved Decisions

### Format scope and rollout

- v1 geometry import formats: `stl` only.
- v1.1 formats: `step`/`stp` (CAD interchange) as the second major format.
- v1.2 formats: `obj`, `ply`.
- v1.3 formats: `gltf`/`glb` static meshes only.
- Out of scope until later: animation, skinning, morph targets, scene graph animation clips.

Rationale:

- `STEP` is prioritized over additional mesh formats to reduce friction for engineering teams with CAD-native workflows.
- Mesh-only expansion continues after CAD interchange support is stable.

### Normalization policy

- Default import path applies safe normalization:
  - dedupe exact duplicate vertices,
  - remove degenerate triangles,
  - normalize winding where recoverable,
  - preserve original source metadata and import diagnostics.
- Any topology-changing repair beyond safe normalization is opt-in and must emit a new geometry revision.

### Capacity limits (initial)

- Desktop profile hard limits:
  - max vertices: 8,000,000
  - max elements/faces: 16,000,000
  - max source file size: 1.5 GiB
- Server profile hard limits:
  - max vertices: 20,000,000
  - max elements/faces: 40,000,000
  - max source file size: 4 GiB
- Limits are configurable by deployment profile but must be validated pre-import.

### Solver backend scope (initial)

- `analysis-fea` v1 backend: linear static structural solve only.
- Supported execution paths in v1:
  - CPU direct solve,
  - CPU iterative solve,
  - optional GPU iterative solve where acceleration backend supports required sparse ops.

### Result post-processing placement

- Field generation and principal/derived metrics run in solver/post layer.
- Projection to display mesh happens in runtime post stage and records projection method in provenance.
- Final display conversion may happen on host for viewer transfer, but canonical result fields remain backend-neutral.

## Canonical Schemas (Normative)

### `GeometryAsset` (minimum required fields)

```json
{
  "geometryId": "geo_...",
  "source": {
    "path": "/models/part.stl",
    "sha256": "...",
    "importerVersion": "stl/v1"
  },
  "sourceGeometry": {
    "kind": "mesh",
    "assembly": null,
    "materialEvidence": []
  },
  "tessellationProfile": {
    "profileId": "default-v1",
    "chordTolerance": null,
    "angleToleranceDeg": null,
    "healingMode": "safe"
  },
  "units": "meter",
  "revision": 1,
  "meshes": [
    {
      "meshId": "mesh_...",
      "kind": "surface",
      "vertexCount": 12345,
      "elementCount": 24680
    }
  ],
  "regions": [],
  "diagnostics": []
}
```

### Material ingestion semantics

- Imported material data is treated as **evidence**, not guaranteed solver-ready truth.
- Material evidence must include:
  - source field/key,
  - normalized target field,
  - confidence (`high|medium|low`),
  - conversion assumptions,
  - unit basis.
- Analysis validation may require user confirmation/override before solve.

### `AnalysisModel` (minimum required fields)

```json
{
  "analysisModelId": "am_...",
  "geometryId": "geo_...",
  "geometryRevision": 1,
  "analysisType": "linear_static_structural",
  "materials": [],
  "boundaryConditions": [],
  "loads": [],
  "steps": [
    { "stepId": "step_1", "kind": "static" }
  ]
}
```

### `AnalysisRunProvenance` (required for persisted runs)

```json
{
  "runId": "run_...",
  "geometryId": "geo_...",
  "geometryRevision": 1,
  "analysisModelHash": "sha256:...",
  "opVersion": "analysis.run/v1",
  "backend": "cpu",
  "deviceId": "cpu:0",
  "precision": "fp64",
  "deterministic": true,
  "fallbackEvents": []
}
```

## Numeric Tolerance Policy

- CPU vs CPU deterministic replay: exact match for topology + metadata; numeric fields within `1e-12` relative tolerance.
- CPU vs GPU parity (fp64 paths): `1e-8` relative tolerance.
- CPU vs GPU parity (mixed/fp32 paths): `1e-5` relative tolerance.
- Any tolerance override must be tied to solver/backend profile and recorded in test fixtures.

## Fixture Matrix (Required)

Maintain fixture corpus with expected diagnostics and solve behavior:

- `tiny-manifold.stl` (happy path)
- `degenerate-faces.stl` (repairable)
- `non-manifold.stl` (warning path)
- `disconnected-bodies.stl` (region segmentation)
- `large-mesh-threshold.stl` (capacity guard)
- `unit-sensitive-case.stl` (unit validation)

Each fixture must define:

- expected import diagnostics,
- expected geometry stats,
- expected validation outcome,
- expected solve status (if applicable).

### Governance

- Benchmark baselines have explicit owners.
- Any baseline update requires:
  - linked justification,
  - before/after metrics,
  - reviewer sign-off.

## Migration Playbook (Operation Versions)

For any contract change:

1. Add new versioned operation schema (`.../vN+1`).
2. Keep previous version fully functional through deprecation window.
3. Add migration tests and fixture snapshots for both versions.
4. Update docs and provenance recorder to emit latest version.
5. Remove legacy version only after explicit removal milestone.

## Implementation Manifest (Unambiguous)

### `runmat-geometry-core`

Create files:

- `crates/runmat-geometry/core/src/lib.rs`
- `crates/runmat-geometry/core/src/model/geometry.rs`
- `crates/runmat-geometry/core/src/model/mesh.rs`
- `crates/runmat-geometry/core/src/model/topology.rs`
- `crates/runmat-geometry/core/src/model/regions.rs`
- `crates/runmat-geometry/core/src/model/field.rs`
- `crates/runmat-geometry/core/src/model/source_geometry.rs`
- `crates/runmat-geometry/core/src/model/tessellation_profile.rs`
- `crates/runmat-geometry/core/src/model/material_evidence.rs`
- `crates/runmat-geometry/core/src/model/assembly.rs`
- `crates/runmat-geometry/core/src/selection/entity_ref.rs`
- `crates/runmat-geometry/core/src/diagnostics/mod.rs`

Required tests:

- identity stability within revision,
- serialization round-trip,
- unit metadata presence validation.

### `runmat-geometry-io`

Create files:

- `crates/runmat-geometry/io/src/lib.rs`
- `crates/runmat-geometry/io/src/sniff.rs`
- `crates/runmat-geometry/io/src/import/stl.rs`
- `crates/runmat-geometry/io/src/import/step.rs`
- `crates/runmat-geometry/io/src/normalize.rs`
- `crates/runmat-geometry/io/src/report.rs`

### CAD interop placement (locked)

CAD interop implementation is placed inside geometry IO:

- `crates/runmat-geometry/io/src/cad/*`

This path must preserve the same contracts:

- `AssemblyNode` graph,
- `TessellationProfile` input/output,
- `MaterialEvidence` extraction payloads.

Required tests:

- stl import happy path,
- degenerate cleanup path,
- capacity guard rejection,
- deterministic import output hash.

### `runmat-geometry-ops`

Create files:

- `crates/runmat-geometry/ops/src/lib.rs`
- `crates/runmat-geometry/ops/src/bounds.rs`
- `crates/runmat-geometry/ops/src/stats.rs`
- `crates/runmat-geometry/ops/src/quality.rs`
- `crates/runmat-geometry/ops/src/queries.rs`

Required tests:

- bounds correctness,
- quality diagnostics for malformed fixture,
- query/entity selection determinism.

### `runmat-analysis-core`

Create files:

- `crates/runmat-analysis/core/src/lib.rs`
- `crates/runmat-analysis/core/src/problem/model.rs`
- `crates/runmat-analysis/core/src/problem/materials.rs`
- `crates/runmat-analysis/core/src/problem/bc.rs`
- `crates/runmat-analysis/core/src/problem/loads.rs`
- `crates/runmat-analysis/core/src/problem/steps.rs`
- `crates/runmat-analysis/core/src/validate.rs`

Required tests:

- missing material/BC/load validation failures,
- unit/frame mismatch rejection,
- valid model acceptance.

### `runmat-analysis-fea`

Create files:

- `crates/runmat-analysis/fea/src/lib.rs`
- `crates/runmat-analysis/fea/src/assembly/mod.rs`
- `crates/runmat-analysis/fea/src/solve/linear.rs`
- `crates/runmat-analysis/fea/src/post/fields.rs`
- `crates/runmat-analysis/fea/src/diagnostics/mod.rs`

Required tests:

- canonical cantilever benchmark (regression fixture),
- convergence diagnostics emission,
- cpu/gpu parity (where backend present).

## Success Criteria

- Users can open and interact with common 3D formats via first-class viewer.
- Same geometry semantics are available through runtime function contracts.
- Tool integrations can call geometry/analysis operations without bespoke adapter logic.
- Storage/run artifacts preserve reproducible lineage for analysis outputs.
- GPU residency provides measurable speedups for large analysis runs with stable fallbacks.
- Architecture supports incremental expansion toward broad multiphysics and HPC domains without contract churn.

## Handover Checklist

For maintainers onboarding mid-project, verify:

1. Operation contracts and versions are current and documented.
2. Geometry identity/remap semantics are covered by tests.
3. Unit/frame validation is enforced pre-solve.
4. Deterministic mode and provenance capture are wired for all backends.
5. Failure-mode diagnostics are visible in run summaries and telemetry.
6. Fixture matrix is passing for import/ops/analysis layers.
7. Contract migration playbook is followed for any op schema change.

## Execution Plan (Kickoff)

### Track A: Geometry foundation (OSS)

1. Scaffold `runmat-geometry/core` with canonical types and invariants.
2. Scaffold `runmat-geometry/io` with sniffing + STL importer + CAD module namespace at `io/src/cad`.
3. Add `runmat-geometry/ops` with bounds/stats/quality/query primitives.
4. Wire runtime operation stubs (`geometry.inspect`, `geometry.load`, `geometry.compute_stats`).

### Track B: Analysis foundation (OSS)

1. Scaffold `runmat-analysis/core` model + validation.
2. Scaffold `runmat-analysis/fea` linear static pipeline skeleton.
3. Add typed error taxonomy and operation envelopes across geometry/analysis ops.

### Track C: Reliability and governance (OSS)

1. Stand up fixture corpus and deterministic replay tests.
2. Add CPU/GPU parity harness with tolerance policy checks.
3. Add contract conformance tests for operation versions.
4. Emit machine-readable benchmark/conformance artifacts and fail CI on gate regressions.
5. Upload benchmark/conformance artifacts from CI for regression triage.
6. Support optional baseline drift enforcement from prior benchmark artifacts.

## Progress Log (OSS)

- 2026-03-06: Added `analysis.run_modal/v1` operation contract scaffold with typed gating: explicit model-shape validation (`ANALYSIS_RUN_MODAL_INVALID_MODEL` when no modal step) and explicit unsupported solver-path contract (`ANALYSIS_RUN_MODAL_UNSUPPORTED`) while modal execution implementation is pending.
- 2026-03-06: Upgraded `analysis.run_modal/v1` from unsupported scaffold to a deterministic placeholder execution path that reuses linear-static infrastructure, emits explicit `FEA_MODAL_PLACEHOLDER` diagnostics + `ModalPlaceholder` quality reason, and forces degraded/non-publishable status until native modal solver implementation lands.
- 2026-03-06: Added first-class modal payload schema (`modal_results`) to analysis run/results contracts with placeholder eigenvalue/mode-shape data (`eigenvalues_hz`, `mode_shapes`) so modal outputs are forward-compatible before native eigen-solver rollout.
- 2026-03-06: Added `analysis.results` modal query controls (`include_modal_results`, `mode_indices`) with typed out-of-range mode mapping (`ANALYSIS_RESULTS_MODE_NOT_FOUND`) so modal payload retrieval can be bounded/filtered as eigen-solve output sizes grow.
- 2026-03-06: Added `analysis.create_model/v1` runtime operation contract with typed intent validation and geometry precondition checks, producing a validated baseline linear-static analysis model from geometry metadata (`geometry_id`, revision, units) and covered by unit + operation-contract tests.
- 2026-03-06: Extended `analysis.create_model/v1` intent schema with explicit profile selection (`linear_static_structural`) so future create-model templates (modal/transient/nonlinear) can be added without changing operation shape, and wired profile-driven template selection in runtime model synthesis.
- 2026-03-06: Implemented `analysis.create_model/v1` modal template support (`modal_structural`) with profile-driven baseline model synthesis (modal step + default zero-body-force seed load), preserving operation shape and validation guarantees.
- 2026-03-06: Kept forward scaffolding for `transient_structural` and `nonlinear_structural` with explicit typed unsupported contract response (`ANALYSIS_CREATE_MODEL_PROFILE_UNSUPPORTED`) to lock non-breaking profile expansion semantics.
- 2026-03-06: Added versioned runtime contract placeholder for `geometry.capture_view/v1` with explicit typed unsupported-capability mapping (`GEOMETRY_CAPTURE_UNSUPPORTED`) and contract/unit test coverage to keep the unified geometry operation surface stable while renderer wiring remains pending.
- 2026-03-06: Wired `geometry.capture_view/v1` behind a thread-scoped runtime capture adapter seam (`GeometryViewCaptureAdapter` + guard), added typed spec validation (`GEOMETRY_CAPTURE_INVALID_SPEC`) and backend-failure mapping (`GEOMETRY_CAPTURE_BACKEND_FAILED`), and retained explicit unsupported fallback when no adapter is installed.
- 2026-03-06: Added a runtime-integrated default geometry capture renderer for SVG snapshots (`runtime-svg-summary` adapter) so `geometry.capture_view/v1` can produce deterministic summary-view captures without external UI state, while non-SVG formats continue to report typed unsupported capability errors unless a custom adapter is installed.
- 2026-03-06: Added runtime geometry operation contracts for `geometry.list_regions/v1` and `geometry.query_entities/v1` (with typed validation errors such as `GEOMETRY_REGION_NOT_FOUND` / `GEOMETRY_QUERY_INVALID_LIMIT`), plus unit/integration contract coverage in the new `runmat-runtime/src/geometry/` module layout.
- 2026-03-06: Refactored runtime geometry module from single-file layout to directory layout (`runmat-runtime/src/geometry/{mod,tests}.rs`) to keep operation implementation and test growth maintainable while preserving existing geometry operation contracts.
- 2026-03-06: Implemented STEP import MVP in `runmat-geometry-io` through `io/src/cad` integration: STEP payloads now parse basic CAD metadata (`FILE_NAME`, `PRODUCT`, material tokens), emit CAD-kind source geometry with assembly/regions/material evidence, and are covered by deterministic import + runtime geometry operation tests.
- 2026-03-06: Added runtime quality policy and reason contracts for analysis runs/results (`quality_policy`, `quality_reasons`) with policy-dependent publishability behavior (`strict`, `balanced`, `exploratory`), and aligned runtime tests/conformance harness defaults to the explicit balanced policy.
- 2026-03-06: Added explicit runtime tests for quality-policy divergence: `balanced` remains publishable when core quality gates pass but quality reasons exist (field-promotion fallback), while `strict` degrades publishability under the same conditions.
- 2026-03-06: Added operation-contract coverage that `strict` policy quality outcomes (`run_status`, `publishable`, `quality_reasons`, `provenance.quality_policy`) propagate consistently through `analysis.run_linear_static`, direct `analysis.results`, and `analysis.results` by `run_id`.
- 2026-03-06: Added operation-contract divergence test for identical GPU field-promotion fallback conditions (forced upload failure): `balanced` remains publishable while `strict` is degraded, with matching core gates and shared `FieldPromotionFallback` reason.
- 2026-03-06: Extended divergence coverage to result retrieval endpoints for identical field-promotion fallback runs, asserting `balanced` vs `strict` publishability/status differences are preserved in both direct `analysis.results` and `analysis.results` by `run_id` responses.
- 2026-03-06: Added material-assignment confidence modeling to analysis domain (`material_assignments` + confidence tiers), emitted trust-tier conflict diagnostics from FEA runs, and validated degraded publishability behavior for low-confidence mismatches via multi-material fixture coverage in runtime contract/harness tests.
- 2026-03-06: Added multi-material assembly fixture coverage (`MultiMaterialAssembly`) with mixed material set + heterogeneous load kinds, validated distinct response behavior in FEA tests, and expanded benchmark harness with provider-backed multi-material residency/sync/ratio gates.
- 2026-03-06: Expanded scale validation with `CantileverLargeLoadSweep` fixture (512 load cases) across FEA + benchmark harness, and added stricter provider-backed residency gates for the extra-large case (`min_solver_device_apply_k_ratio=1.0`, bounded host-sync budget) to verify higher-DOF behavior under the runtime-tensor path.
- 2026-03-06: Tightened GPU-provider conformance gates in benchmark harness by raising minimum device operator application ratio to `1.0` and lowering maximum solver host-sync budget to `32` for provider-backed fixtures, confirming stricter residency expectations pass under current runtime-tensor path.
- 2026-03-06: Added provider-capability-aware harness provider implementation for benchmark conformance (supports elementwise/reduction/gather/scatter operations), raised device-operator telemetry gates for provider-backed fixtures (`min_solver_device_apply_k_ratio`), and validated ratio-driven conformance checks with upgraded runtime-tensor coverage.
- 2026-03-06: Added provider-capability telemetry for device operator application progress (`solver_device_apply_k_ratio`) and integrated fixture-level minimum-ratio + sync-budget gating into benchmark conformance harness, alongside runtime-tensor preconditioner/workspace reuse improvements.
- 2026-03-06: Added runtime-tensor scratch workspace reuse for preconditioner sweeps (`RuntimeTensorWorkspace` with reusable device buffers for `y/z`) and switched preconditioner update loops to in-place scatter updates, reducing per-iteration allocation/free churn in GPU solve loops.
- 2026-03-06: Added device-resident preconditioner application paths in runtime-tensor solver for both Jacobi and ILU(0)-style modes (`apply_preconditioner_device` + iterative device sweeps for ILU approximation), removing additional host-dependent preconditioner work from the GPU solve loop.
- 2026-03-06: Reduced runtime-tensor PCG host synchronization overhead by removing the initial `z -> p` host roundtrip (device-side initialization) and including explicit fallback-download/output-download accounting in `solver_host_sync_count`, improving fidelity of residency telemetry and sync-budget gates.
- 2026-03-06: Added sync-budget conformance controls in benchmark harness (`max_solver_host_sync_count` per fixture) and wired `solver_host_sync_count` checks into GPU fixture gating, so host-sync regression thresholds are enforced alongside parity/residency/error contracts.
- 2026-03-06: Hardened runtime-tensor backend selection semantics: GPU solve requests now choose `runtime_tensor` only when an acceleration provider is available, otherwise explicitly fall back to `cpu_reference` with typed fallback telemetry (`SOLVER_BACKEND_FALLBACK`), while provenance reports the actual solver backend used.
- 2026-03-06: Added solver host-sync accounting (`solver_host_sync_count`) into solve/runtime provenance and reduced duplicate scalar syncs in runtime-tensor PCG by reusing residual reductions; this makes GPU residency progress measurable in contracts/harness artifacts.
- 2026-03-06: Reduced host round-trips in runtime-tensor PCG by adding device-side operator application composition (`apply_k_device`) using provider gather/elementwise/reduction primitives, while retaining guarded fallback to host operator application when provider capabilities are missing.
- 2026-03-06: Added initial runtime-tensor PCG solver path (`solve/runtime_tensor_solver.rs`) that keeps iterative vector algebra on provider-backed tensors with limited host interaction for operator application/scalars, and wired backend dispatch to prefer runtime-tensor solve when available while preserving guarded per-op fallback behavior.
- 2026-03-06: Added initial `runtime_tensor` linear-algebra backend implementation in FEA solve backend layer (provider-backed vector ops via `runmat-accelerate-api` with guarded CPU-reference fallback per operation), and updated runtime GPU solve mapping to request runtime-tensor backend through the new backend-kind interface.
- 2026-03-06: Added explicit linear algebra backend seam selection (`LinearAlgebraBackendKind`) in FEA options/runtime mapping, with trait-based backend dispatch and explicit solver-backend provenance fields to support backend transition without contract churn.
- 2026-03-06: Added solver math backend seam in FEA (`solve/backend/{linear_algebra,cpu_reference}.rs`) and routed linear solve through backend trait operations, so CPU reference kernels are isolated implementation details and future runtime tensor backend integration can slot in without changing solver contracts.
- 2026-03-06: Implemented second preconditioner path with tridiagonal ILU(0)-style preconditioning (selectable via `preconditioner_mode=ilu`), upgraded solve selection to pass requested mode into FEA internals, and retained explicit fallback telemetry only for unsupported requested modes (currently `amg -> jacobi`).
- 2026-03-06: Introduced SPD preconditioner abstraction in FEA solve path (`solve/preconditioner.rs`) with pluggable interface and Jacobi implementation, surfaced solver/preconditioner selection in runtime provenance, and added explicit preconditioner fallback telemetry (`SOLVER_PRECONDITIONER_FALLBACK`) when unsupported modes are requested.
- 2026-03-06: Added non-diagonal stiffness coupling to matrix-free operator application (`Kx` tridiagonal-style neighbor coupling with constrained-DOF treatment), upgraded iterative solve to PCG + Jacobi preconditioning, and added tighter physics-oriented fixture checks (cantilever displacement/stress tolerances plus load-sweep magnitude scaling assertions).
- 2026-03-06: Advanced solver depth with preconditioned matrix-free iterative solve details: upgraded linear path to PCG + Jacobi preconditioner diagnostics (`FEA_SOLVER_METHOD`), added large-load fixture (`CantileverLoadSweep`) for scale-oriented validation, and expanded benchmark/conformance manifest with the larger GPU-provider fixture.
- 2026-03-06: Upgraded `runmat-analysis-fea` from placeholder solve to a matrix-free operator-backed linear static path: added operator API module (`apply_k/apply_m/apply_c`), assembly now produces an `OperatorSystem` with rhs/diagonals/constraints, linear solve uses iterative conjugate-gradient over operator application, and post-fields derive from solved displacement vectors (with existing runtime contracts preserved).
- 2026-03-06: Added optional baseline drift comparison in benchmark harness (`RUNMAT_ANALYSIS_BASELINE_PATH`, `RUNMAT_ANALYSIS_ENFORCE_BASELINE`, `RUNMAT_ANALYSIS_MAX_SLOWDOWN_RATIO`) to compare CPU/GPU timing ratios against a prior artifact and enforce slowdown gates when requested.
- 2026-03-06: Added run-artifact persistence boundary for analysis runtime (`AnalysisArtifactStore` adapter with in-memory default + filesystem adapter), introduced `run_id` on `analysis.run_linear_static` responses, and implemented `analysis.results` retrieval by run id with typed missing-lineage error (`ANALYSIS_RESULTS_RUN_NOT_FOUND`).
- 2026-03-06: Implemented `analysis.results/v1` runtime operation with typed field filtering and metadata response (`fields`, `diagnostics`, status/provenance/summary), added typed unknown-field error mapping (`ANALYSIS_RESULTS_FIELD_NOT_FOUND`), and integrated results-op conformance checks into unit, contract, and benchmark harness tests.
- 2026-03-06: Wired CI artifact publishing for analysis benchmark reports by uploading `target/runmat-analysis-artifacts/*.json` from `ci.yml` after test execution, enabling per-matrix run triage.
- 2026-03-06: Added unified benchmark/conformance harness test (`runmat-runtime/tests/analysis_benchmark_conformance.rs`) with fixture manifest schema, CPU/GPU runner, machine-readable JSON artifact output (`target/runmat-analysis-artifacts/analysis_benchmark_report.json` by default), and hard gates for parity tolerance, contract error/version expectations, fallback schema, and GPU residency expectations.
- 2026-03-06: Added fallback-event schema conformance checks in runtime operation contract tests, validating stable parsable format (`category:stage:reason`) for GPU fallback telemetry events.
- 2026-03-06: Extended operation contract tests to cover provider-upload failure fallback semantics (`BACKEND_UPLOAD_FAILED`) for GPU-targeted analysis runs, asserting host field retention and explicit fallback provenance events.
- 2026-03-06: Refactored runtime analysis module into directory layout (`runmat-runtime/src/analysis/{mod,contracts,promotion,tests}.rs`) to separate operation surface, contract types, device-ref promotion logic, and tests without changing operation semantics.
- 2026-03-06: Extended runtime operation contract integration tests to lock GPU field residency behavior: explicit fallback-event contract when no provider is present, and `AnalysisFieldValues::DeviceRef` contract when provider-backed promotion is available.
- 2026-03-06: Runtime analysis now opportunistically promotes GPU solve output fields to `AnalysisFieldValues::DeviceRef` via `runmat-accelerate-api` provider upload hooks, and records explicit fallback events (`BACKEND_NO_PROVIDER`, `BACKEND_UPLOAD_FAILED`) when promotion cannot happen.
- 2026-03-06: Added explicit runtime solve options for `analysis.run_linear_static` (`deterministic_mode`, `precision_mode`) and extended contract tests to assert deterministic replay stability plus provenance backend/precision recording across CPU/GPU runs.
- 2026-03-06: Extended runtime contract conformance tests for `analysis.validate` mismatch scenarios to assert exact typed envelope mapping for `ANALYSIS_VALIDATION_UNIT_MISMATCH` and `ANALYSIS_VALIDATION_FRAME_MISMATCH` (including structured mismatch context fields).
- 2026-03-06: Expanded Track C fixture coverage: added invalid analysis fixtures (`MissingMaterials`, `MissingLoads`) in `runmat-analysis-fea::fixtures`, added fixture rejection tests in FEA, and extended runtime contract conformance tests to assert failure envelope mapping (`SOLVER_MODEL_INVALID`) and field-contract shape (`AnalysisFieldValues`).
- 2026-03-06: Replaced raw analysis result vectors in FEA contracts with `AnalysisField` abstraction (`host_f64 | device_ref`) in `runmat-analysis-core`, and updated FEA/runtime tests to consume fields through the domain boundary to avoid throwaway contract types.
- 2026-03-06: Track C kickoff. Added initial fixture corpus wiring in `runmat-analysis-fea` (`fixtures::FixtureId` + canonical cantilever model), deterministic replay test coverage for repeated solve stability, CPU/GPU parity tolerance policy helper (`ParityTolerance`) with parity tests, and runtime contract conformance integration tests for operation/version/error-code envelopes (`--test operation_contracts`).
- 2026-03-06: Added solve gating and provenance fields to runtime analysis envelopes (`model_validity`, `solver_convergence`, `result_quality`, `run_status`, `publishable`, and backend/precision/determinism/fallback provenance core) so publishability decisions are explicit and machine-readable.
- 2026-03-06: Runtime operation envelopes added for geometry and analysis. Introduced typed `OperationEnvelope`/`OperationErrorEnvelope`, implemented `analysis.validate/v1` and `analysis.run_linear_static/v1` runtime ops, upgraded geometry ops to `geometry.inspect/load/compute_stats` envelopes, and aligned machine error families (`GEOMETRY_*`, `ANALYSIS_*`, `SOLVER_*`, `CAPACITY_*`) with tests.
- 2026-03-06: Track B started. Added `runmat-analysis-core` (`problem/*` + `validate`) and `runmat-analysis-fea` (`assembly`, `solve/linear`, `post/fields`, `diagnostics`) with workspace wiring and scaffold tests for validation failures, unit/frame mismatch rejection, canonical cantilever run, convergence diagnostic emission, and CPU/GPU parity behavior.
- 2026-03-06: Track A scaffolding validated end-to-end. Fixed `runmat-geometry-io` compile regressions (`import_stl` module path and test `ImportResult` import), then passed `cargo test -p runmat-geometry-io -p runmat-geometry-ops` and runtime geometry smoke tests via `cargo test -p runmat-runtime geometry::tests::`.
- 2026-03-06: Track A started. Added new crate `runmat-geometry-core` at `crates/runmat-geometry/core`, wired workspace membership/dependency, scaffolded canonical domain modules (`model`, `selection`, `diagnostics`), and added initial invariant tests (identity stability within revision, JSON round-trip, unit metadata validation).
- 2026-03-05: Architecture finalized for geometry + analysis stack; docs now include typed error model, invariants, implementation manifest, CAD/STEP prioritization, and crate layout under `crates/runmat-geometry/*` and `crates/runmat-analysis/*`.
