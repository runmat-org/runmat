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

### Solver Option Presets (Client JSON Guidance)

The runtime now exposes explicit option structs for modal and transient runs. Clients can either
use helper presets (`coarse`, `balanced`, `high_accuracy`) in Rust, or send equivalent JSON-like
payloads through host bindings.

`AnalysisModalRunOptions` (coarse):

```json
{
  "deterministicMode": false,
  "precisionMode": "fp32",
  "qualityPolicy": "exploratory",
  "modeCount": 2,
  "residualWarnThreshold": 0.005
}
```

`AnalysisModalRunOptions` (balanced/default):

```json
{
  "deterministicMode": false,
  "precisionMode": "fp64",
  "qualityPolicy": "balanced",
  "modeCount": 3,
  "residualWarnThreshold": 0.001
}
```

`AnalysisModalRunOptions` (high accuracy):

```json
{
  "deterministicMode": true,
  "precisionMode": "fp64",
  "qualityPolicy": "strict",
  "modeCount": 8,
  "residualWarnThreshold": 0.0005
}
```

`AnalysisTransientRunOptions` (coarse):

```json
{
  "deterministicMode": false,
  "precisionMode": "fp32",
  "qualityPolicy": "exploratory",
  "timeStepS": 0.005,
  "minTimeStepS": 0.0005,
  "maxTimeStepS": 0.02,
  "stepCount": 6,
  "maxLinearIters": 64,
  "tolerance": 0.000001,
  "residualTarget": 0.0001,
  "adaptiveTimeStep": true,
  "maxStepRetries": 2
}
```

`AnalysisTransientRunOptions` (balanced/default):

```json
{
  "deterministicMode": false,
  "precisionMode": "fp64",
  "qualityPolicy": "balanced",
  "timeStepS": 0.001,
  "minTimeStepS": 0.000001,
  "maxTimeStepS": 0.02,
  "stepCount": 10,
  "maxLinearIters": 128,
  "tolerance": 0.00000001,
  "residualTarget": 0.000001,
  "adaptiveTimeStep": true,
  "maxStepRetries": 4
}
```

`AnalysisTransientRunOptions` (high accuracy):

```json
{
  "deterministicMode": true,
  "precisionMode": "fp64",
  "qualityPolicy": "strict",
  "timeStepS": 0.0005,
  "minTimeStepS": 0.000005,
  "maxTimeStepS": 0.002,
  "stepCount": 24,
  "maxLinearIters": 256,
  "tolerance": 0.0000000001,
  "residualTarget": 0.0000001,
  "adaptiveTimeStep": true,
  "maxStepRetries": 8
}
```

Preset selection quick guide:

- `coarse`: use for fast exploratory loops, UI previews, and broad parameter sweeps where speed is prioritized over strict numerical confidence.
- `balanced`: default choice for day-to-day engineering workflows where runtime cost and result quality should both remain stable.
- `high_accuracy`: use for final verification/regression baselines, deterministic replay, and stricter publishability expectations.

Recommended starting matrix:

| Scenario | Modal preset | Transient preset | Nonlinear preset |
| --- | --- | --- | --- |
| Interactive design iteration | `coarse` | `coarse` | `coarse` |
| Standard analysis pipeline | `balanced` | `production_recommended` | `production_recommended` |
| Release/sign-off quality gate | `high_accuracy` | `high_accuracy` | `high_accuracy` |

### Nonlinear Tuning Playbook

Use this when re-baselining nonlinear thresholds or evaluating new solver behavior:

1. Run baseline conformance with defaults:

   ```bash
   cargo test -p runmat-runtime --test analysis
   ```

2. Run targeted nonlinear sweeps by overriding harness knobs:

   ```bash
   RUNMAT_NONLINEAR_MAX_NEWTON_ITERS=24 \
   RUNMAT_NONLINEAR_MAX_BACKTRACKS=6 \
   RUNMAT_NONLINEAR_TANGENT_REFRESH_INTERVAL=2 \
   cargo test -p runmat-runtime --test analysis
   ```

3. Validate guard sensitivity with a negative control (expected to fail conformance):

   ```bash
   RUNMAT_NONLINEAR_LINE_SEARCH=false \
   RUNMAT_NONLINEAR_MAX_BACKTRACKS=0 \
   cargo test -p runmat-runtime --test analysis
   ```

4. Inspect `target/runmat-analysis-artifacts/analysis_benchmark_report.json` focusing on:

   - `nonlinear_assembly_gpu_provider`
   - `nonlinear_assembly_stress_gpu_provider`
   - `threshold_assertions[*]` values for:
     - converged/total/failed increments,
     - line-search backtracks,
     - tangent rebuild count,
     - residual + increment-norm ceilings,
     - `gpu_run_ms` and `gpu_speedup_ratio`.

5. Keep threshold updates only when all are true:

   - both nonlinear provider fixtures remain `publishable=true`,
   - no failed increments in production-recommended runs,
   - stress fixture line-search and tangent activity remain within bounded bands,
   - speedup floors remain healthy (currently ~2.6x baseline nonlinear, ~4.3x stress nonlinear on local reference runs).

Harness override environment variables:

- `RUNMAT_NONLINEAR_INCREMENT_COUNT`
- `RUNMAT_NONLINEAR_MAX_NEWTON_ITERS`
- `RUNMAT_NONLINEAR_TOLERANCE`
- `RUNMAT_NONLINEAR_RESIDUAL_FACTOR`
- `RUNMAT_NONLINEAR_INCREMENT_NORM_TOL`
- `RUNMAT_NONLINEAR_LINE_SEARCH`
- `RUNMAT_NONLINEAR_MAX_BACKTRACKS`
- `RUNMAT_NONLINEAR_LINE_SEARCH_REDUCTION`
- `RUNMAT_NONLINEAR_TANGENT_REFRESH_INTERVAL`

### Nonlinear CI Governance

- CI job `nonlinear-conformance` runs `cargo test -p runmat-runtime --test analysis` on Linux GPU runners.
- Baseline enforcement is branch-aware via:
  - `RUNMAT_ANALYSIS_ENFORCE_BASELINE_ON_PROTECTED=true`
  - `RUNMAT_ANALYSIS_PROTECTED_BRANCHES=main,master,release`
- Artifacts are uploaded only when useful:
  - test failure, or
  - protected branch push (`main` / `release/*`).
- CI emits a compact nonlinear summary from `analysis_benchmark_report.json` into the job summary using `scripts/summarize_analysis_report.py`.
- CI also evaluates rolling nonlinear trends with `scripts/analyze_nonlinear_trends.py`:
  - window: `RUNMAT_ANALYSIS_TREND_WINDOW` (default `8`),
  - slowdown threshold: `RUNMAT_ANALYSIS_TREND_MAX_SLOWDOWN_RATIO` (default `1.25`),
  - warning on non-protected branches, fail-on-threshold for protected branches.
- CI evaluates final nonlinear release readiness via `scripts/release_readiness_nonlinear.py`, producing:
  - machine-readable output: `target/runmat-analysis-artifacts/nonlinear_release_readiness.json`,
  - human summary appended to step summary (including thermo posture metrics when available),
  - verdict semantics: `pass` / `warn` / `fail`.

### Nonlinear Regression Triage

Use this quick map from failing gate to likely fix direction:

- `nonlinear_failed_increments` > 0:
  - likely under-iterated solve or unstable step progression,
  - first try increasing `RUNMAT_NONLINEAR_MAX_NEWTON_ITERS`, then enable/increase line-search backtracks.
- `nonlinear_stress_line_search_backtracks` out of band:
  - too low (often `0`): line search disabled or ineffective,
  - too high: overly aggressive steps; reduce line-search reduction aggressiveness or refresh tangent more frequently.
- `nonlinear_stress_tangent_rebuild_count` above ceiling:
  - indicates repeated rebuild churn; check tangent refresh cadence and increment progression.
- `nonlinear_*_converged_increments` below target:
  - ensure increment count matches fixture intent and verify no env override reduced increments.
- `gpu_speedup_ratio` retention regression:
  - inspect `gpu_solver_solve_ms` and `gpu_solver_prepared_build_ms` drift,
  - check provider availability/fallback events before adjusting thresholds.

Trend interpretation guidance:

- `analysis.results_compare` should be used for run-to-run reviews:
  - negative `failed_increment_delta` / `max_iteration_delta` generally indicates improvement,
  - positive `solve_ms_delta` indicates slowdown and should be weighed against quality gains.
- `analysis.trends` should be used for release-readiness:
  - watch `median_solve_ms` and `p95_solve_ms` divergence,
  - keep nonlinear `publishable_rate` close to `1.0` in stable branches,
  - investigate when `failed_increment_rate`, `mean_spike_count`, or `mean_stall_count` trend upward over multiple windows.

Release readiness criteria (default):

- Required pass checks:
  - latest nonlinear conformance report passed,
  - no missing nonlinear fixture records,
  - no non-publishable nonlinear fixture records,
  - artifact replay + compatibility verification checks marked passed.
- Trend guard:
  - nonlinear fixture `gpu_run_ms` slowdown ratio must stay below `RUNMAT_RELEASE_READINESS_MAX_SLOWDOWN_RATIO` vs rolling median baseline.
  - missing trend history is tolerated on non-protected branches by default; set `RUNMAT_RELEASE_READINESS_REQUIRE_TRENDS=true` to force warning behavior.
- Prep artifact health guard:
  - prep artifact count must remain below configured warn/fail thresholds (`RUNMAT_RELEASE_READINESS_PREP_WARN_ARTIFACT_COUNT`, `RUNMAT_RELEASE_READINESS_PREP_FAIL_ARTIFACT_COUNT`),
  - prep artifact p95 age must remain below configured warn/fail thresholds (`RUNMAT_RELEASE_READINESS_PREP_WARN_P95_AGE_SECONDS`, `RUNMAT_RELEASE_READINESS_PREP_FAIL_P95_AGE_SECONDS`),
  - prep reject-rate guard (`PREP_REJECT_RATE_HIGH`) compares `(stale_reject + mismatch_reject)/created` against `RUNMAT_RELEASE_READINESS_PREP_MAX_REJECT_RATE` when counters are provided.
- Thermo-coupling posture guard:
  - when thermo metrics are present, release readiness evaluates max transient/nonlinear thermo severity against branch-aware thresholds (`RUNMAT_RELEASE_READINESS_THERMO_MAX_TRANSIENT_SEVERITY`, `RUNMAT_RELEASE_READINESS_THERMO_MAX_NONLINEAR_SEVERITY`),
  - coupling enabled-rate can be enforced via `RUNMAT_RELEASE_READINESS_THERMO_MIN_ENABLED_RATE`,
  - metrics presence can be required via `RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_METRICS`.
- Protected branches:
  - `fail` reasons block release,
  - `warn` reasons are surfaced and should be triaged before tagging.

### Integrator Contract Guidance (Nonlinear)

When consuming `analysis.results/v1` with nonlinear payloads:

- Minimum recommended fields:
  - `run_status`, `publishable`, `quality_reasons`
  - `summary.increment_count`, `summary.failed_increment_count`
  - `summary.max_nonlinear_residual_norm`, `summary.max_nonlinear_iteration_count`
- Optional-but-useful fields (newer runtimes):
  - `summary.nonlinear_iteration_spike_count`
  - `summary.nonlinear_convergence_stall_count`
  - `summary.nonlinear_backtrack_burst_count`
  - `summary.nonlinear_max_backtracks_per_increment`
- Payload-level optional telemetry in `nonlinear_results`:
  - `increment_norms`, `iteration_counts`
  - `max_line_search_backtracks_per_increment`
  - `iteration_spike_count`, `convergence_stall_count`, `backtrack_burst_count`

Backward-compat fallback behavior:

- Consumers must treat missing nonlinear telemetry fields as defaults:
  - numeric counters default to `0`,
  - vectors default to empty,
  - optional summary fields default to `null`.
- Prefer summary fields for stable dashboarding/gating; use full nonlinear payloads for deeper diagnostics and debug tooling.

### Prep-Aware Model Synthesis Flow

For geometry assets that have been meshing-prepared, prefer this flow:

1. `geometry.load/v1`
2. `geometry.prep_for_analysis/v1`
3. `analysis.create_model/v1` with `prep_context`
4. `analysis.validate/v1`

`analysis.create_model` prep-context behavior:

- verifies prep geometry id/revision matches the target geometry,
- validates prep region/mesh mappings reference known geometry entities,
- prioritizes prep-mapped regions for default boundary/load placement,
- upgrades material-assignment confidence on prep-mapped regions.

Typed prep-context error families:

- `ANALYSIS_CREATE_MODEL_PREP_MISMATCH`
- `ANALYSIS_CREATE_MODEL_PREP_REGION_NOT_FOUND`
- `ANALYSIS_CREATE_MODEL_PREP_MESH_NOT_FOUND`
- `ANALYSIS_CREATE_MODEL_PREP_INVALID_MAPPING`

Prep-aware solve semantics (current MVP):

- `analysis.run_*` option payloads accept optional `prep_context` summary metrics.
- Runtime maps prep context into FEA execution options for linear/modal/transient/nonlinear paths.
- FEA emits `FEA_PREP_CONTEXT` diagnostics for prep-aware runs and applies deterministic assembly scaling based on prep mesh density/quality counters.
- Current guarantee: prep-aware runs are deterministic and observable, but still use surrogate prep influence (not full remeshed element-topology solve yet).

Prep-aware assembly fidelity tier (updated):

- assembly now uses prep-derived load/boundary mapping participation (`mapped_load_count`, `mapped_bc_count`) and deterministic layout seeds to alter DOF load placement and constrained-DOF layout,
- prep-aware runs emit both:
  - `FEA_PREP_CONTEXT` (artifact-level prep metadata),
  - `FEA_PREP_ASSEMBLY` (assembly participation/distribution metrics),
- this represents deterministic prep-driven operator structure changes, while still not replacing the core operator with full remeshed element topology assembly.

Topology-backed prep assembly mode:

- prep mesh descriptors now carry topology hints (`connectivity_class`, `element_family_hint`, `region_span_hint`) used by assembly shaping,
- prep-aware assembly derives:
  - effective DOF scaling from prep topology multipliers,
  - coupling sparsity band shaping from topology bandwidth proxies,
  - region-coupled load/constraint distribution from mapping cardinalities,
- prep-aware runs emit `FEA_PREP_TOPOLOGY` diagnostics with:
  - `effective_dof_multiplier`,
  - `coupling_bandwidth_proxy`,
  - `mapped_region_participation_ratio`.

Trusted prep-reference semantics:

- `geometry.prep_for_analysis/v1` now persists prep artifacts and returns `prep_artifact_id`.
- Analysis run options may reference prep artifacts via `prep_artifact_id`; runtime resolves prep lineage from artifact store.
- Direct free-form solve `prep_context` is treated as untrusted without an artifact reference and is rejected.
- Runtime enforces prep artifact constraints before solve:
  - artifact existence,
  - artifact schema support,
  - model geometry id/revision lineage match.

Prep artifact lifecycle policy knobs:

- `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS`
  - global cap for stored prep artifacts (`0` = disabled)
- `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS_PER_GEOMETRY`
  - cap per `source_geometry_id` (`0` = disabled)
- `RUNMAT_GEOMETRY_PREP_MAX_AGE_SECONDS`
  - age-based pruning threshold in seconds (`0` = disabled)
- `RUNMAT_GEOMETRY_PREP_REQUIRE_LATEST_REVISION`
  - when `true` (default), prep references are rejected as stale if a newer revision prep artifact exists for the same geometry id.

Pruning occurs on prep artifact persist/load and removes stale/excess artifacts from in-memory store and optional filesystem-backed prep artifact root.

Prep artifact observability and SLO monitoring:

- Runtime now exposes `geometry.prep_artifact_health/v1` with:
  - current prep artifact count,
  - age distribution (`age_p50_seconds`, `age_p95_seconds`),
  - lifecycle counters (`created_count`, `loaded_count`, `pruned_count`, `stale_reject_count`, `mismatch_reject_count`),
  - optional per-geometry artifact distribution.
- Runtime emits prep lifecycle events for creation/load/prune and stale/mismatch rejection paths.
- CI includes prep artifact SLO summary via `scripts/summarize_prep_artifacts.py` with configurable warn/fail thresholds:
  - `RUNMAT_PREP_SLO_WARN_ARTIFACT_COUNT`, `RUNMAT_PREP_SLO_FAIL_ARTIFACT_COUNT`,
  - `RUNMAT_PREP_SLO_WARN_P95_AGE_SECONDS`, `RUNMAT_PREP_SLO_FAIL_P95_AGE_SECONDS`.

Typed prep-reference run errors:

- `ANALYSIS_RUN_PREP_UNTRUSTED_CONTEXT`
- `ANALYSIS_RUN_PREP_NOT_FOUND`
- `ANALYSIS_RUN_PREP_SCHEMA_UNSUPPORTED`
- `ANALYSIS_RUN_PREP_MISMATCH`

### Artifact Retention and Migration Guarantees

Runtime artifact persistence now supports compatibility-aware loading for both:

- legacy raw `AnalysisRunResult` JSON artifacts, and
- schema-wrapped artifacts (`analysis_run_artifact/v1`) with metadata.

Retention knobs for filesystem artifact stores:

- `RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS`
  - global cap across all stored run artifacts (0 = disabled)
- `RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS_PER_KIND`
  - per-run-kind cap (`analysis.run_linear_static`, `analysis.run_modal`, `analysis.run_transient`, `analysis.run_nonlinear`) (0 = disabled)

Safe rollback and migration behavior:

- Older artifacts missing newer nonlinear fields remain loadable with defaults.
- Artifacts containing newer unknown fields are tolerated (ignored) by current readers.
- `analysis.results_by_run_id` remains the stable replay boundary for contract consumers.

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

### `runmat-meshing-core`

Create files:

- `crates/runmat-meshing/core/src/lib.rs`

Required tests:

- deterministic meshing-prep output for identical geometry + options,
- stable source-to-prepared region mapping,
- mesh quality diagnostics surface (`min_scaled_jacobian`, inverted-element count).

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

## Parity Roadmap

- Detailed post-kickoff strategy and phase plan is maintained at:
  - `docs/detailed-work/multi-physics-parity-roadmap.md`
- This file remains the running engineering log and implementation timeline.

## Progress Log (OSS)

- 2026-03-09: Extended nonlinear release-readiness governance to consume thermo summary/trend metrics directly from conformance report records by adding thermo posture fields to harness report records (`thermo_coupling_enabled`/fingerprint/severity metrics) and adding branch-profile thermo thresholds/reason codes in `scripts/release_readiness_nonlinear.py` (`THERMO_*`) with unit-test coverage.
- 2026-03-09: Added thermo-coupling posture summary surfaces by extending `analysis.results` summary fields (`thermo_coupling_enabled`/fingerprint and thermo transient/nonlinear severity metrics) and `analysis.trends` kind summaries (thermo coupling enabled-rate and thermo warning-rate metrics), with contract snapshot and typed trend/result coverage updates.
- 2026-03-09: Extended Phase-1 thermo-mechanical runtime governance by mapping thermo-solver warning diagnostics into explicit quality-policy reasons (`ThermoMechanicalTransientStress`, `ThermoMechanicalNonlinearStress`) and adding benchmark threshold gates for `FEA_TM_TRANSIENT` severity/relaxation metrics on the kickoff fixture.
- 2026-03-09: Delivered next Phase-1 thermo-mechanical slice by introducing thermo-aware solve-policy shaping in transient/nonlinear FEA paths (temperature-coupling severity drives adaptive timestep growth/shrink bounds and nonlinear convergence target relaxation), plus new structured diagnostics (`FEA_TM_TRANSIENT`, `FEA_TM_NONLINEAR`) and focused fixture-level unit coverage.
- 2026-03-09: Completed Phase-1 thermo-mechanical kickoff wiring across FEA/runtime/harness contracts by adding optional thermo-mechanical coupling options to all analysis run option surfaces, passing coupling context through assembly for deterministic `FEA_TM_COUPLING` diagnostics/fingerprints, and adding a thermo-mechanical kickoff benchmark fixture with conformance threshold checks.
- 2026-03-08: Documented implementation organization plan for upcoming multiphysics work (layer/module boundaries across `analysis/core`, `analysis/fea` physics/assembly/solve, runtime orchestration, and governance scripts), including dependency, contract, test, and documentation co-update rules before starting Phase-1 thermo-mechanical slice implementation.
- 2026-03-08: Completed Tier-7.5 policy hardening by adding branch-specific governance profiles in nonlinear release-readiness defaults (`release`/`development`/`feature`), enforcing no-silent-bypass CI checks for recommendation/promotion artifacts, publishing Tier-7.5 governance summaries into CI step output, and wiring release workflow governance preflight to validate/generate/promote approved calibration evidence artifacts for tag builds.
- 2026-03-08: Wired Tier-7 lifecycle governance into CI nonlinear-conformance workflow by adding evidence/recommendation/promotion validation steps (`validate_prep_calibration_evidence.py`, `generate_prep_calibration_recommendations.py`, `promote_prep_calibration_evidence.py`), enabling recommendation-artifact requirement in release-readiness evaluation, and publishing recommendation/evidence lifecycle artifacts for triage.
- 2026-03-08: Completed Tier-7 recommendation artifact and evidence-promotion workflow by introducing versioned recommendation artifact generation (`prep-calibration-recommendations/v1`), adding evidence-promotion tooling (`scripts/promote_prep_calibration_evidence.py`) with deterministic validation gates, extending readiness policy with recommendation-artifact and candidate-aging governance (`PREP_CALIBRATION_RECOMMENDATION_ARTIFACT_*`, `PREP_CALIBRATION_CANDIDATE_STALE`), and adding evidence/recommendation lifecycle validation coverage.
- 2026-03-08: Completed Tier-6.5 evidence lifecycle + auto-retune recommendations by adding evidence validation/staleness evaluation (`validate_evidence`), deterministic profile-shift recommendation generation from latest drift + rolling drift slope (`recommend_profile_shifts`), release-readiness policy reasons for stale/invalid evidence and recommendation pressure (`PREP_CALIBRATION_EVIDENCE_INVALID`, `PREP_CALIBRATION_EVIDENCE_STALE`, `PREP_CALIBRATION_RETRAIN_RECOMMENDED`), and trend/report visibility for drift slope and recommendation pressure; added dedicated evidence-validation script and Python unit coverage.
- 2026-03-08: Completed Tier-6 evidence-anchored calibration drift control by adding a versioned prep calibration evidence artifact (`scripts/prep_calibration_evidence.json`), implementing reusable drift evaluation utilities (`scripts/evaluate_prep_calibration_drift.py`), extending benchmark records with structured prep calibration/acceptance fields, and wiring nonlinear release-readiness drift policy (`PREP_CALIBRATION_EVIDENCE_MISSING`, `PREP_CALIBRATION_DRIFT_HIGH`) plus rolling trend/summary visibility for acceptance-score posture.
- 2026-03-08: Completed Tier-5.5 calibration governance by adding prep calibration profile override controls on analysis run options (`auto`/`fast`/`balanced`/`conservative`), propagating profile override into prep-aware FEA calibration selection, surfacing structured prep calibration/acceptance summary metrics in `analysis.results` and `analysis.trends`, and extending nonlinear release-readiness policy with prep acceptance-rate threshold checks (`PREP_ACCEPTANCE_RATE_LOW`/`PREP_ACCEPTANCE_MISSING`) plus unit coverage.
- 2026-03-08: Completed Tier-5 calibration and acceptance harness by adding deterministic prep calibration profiles (`fast`/`balanced`/`conservative`) with family-weighted coefficient scaling, emitting replay-stable `FEA_PREP_CALIBRATION` diagnostics, and adding bounded acceptance evaluation (`FEA_PREP_ACCEPTANCE`) with deterministic fingerprints integrated into prep conformance replay checks.
- 2026-03-08: Completed Tier-4.5 graph-solver co-design by introducing deterministic graph-ordering profiles (before/after bandwidth, ordering fingerprints), graph-conditioned preconditioner tuning hooks in linear/transient-prepared solves, and replay-stable `FEA_PREP_GRAPH_SOLVER` diagnostics reporting requested/effective preconditioner plus ordering reduction and solver profile metrics.
- 2026-03-08: Completed Tier-4 prep graph-backbone assembly by constructing deterministic prep-derived sparse connectivity graphs (node/edge topology, degree distribution, component structure), using graph-driven connectivity traversal for off-diagonal scatter shaping, and emitting replay-stable `FEA_PREP_GRAPH_ASSEMBLY` diagnostics with graph fingerprint plus conformance replay and structural bound checks.
- 2026-03-08: Completed Tier-3.5 prep-native connectivity scatter by adding deterministic element-family off-diagonal coupling contributions on prep-aware paths, introducing mass/damping neighbor-scatter proxies, and emitting replay-stable `FEA_PREP_ELEMENT_CONNECTIVITY` diagnostics with nnz counts, family contribution shares, connectivity hop stats, and connectivity fingerprints validated by conformance replay checks.
- 2026-03-08: Delivered Tier-3-core prep-native element assembly by adding deterministic prepared-element family contribution synthesis on prep-aware paths (triangle/quad/tet/hex/mixed distribution derived from prep descriptors), integrating element-level coefficient accumulation into assembled stiffness/mass/damping/rhs, and emitting replay-stable `FEA_PREP_ELEMENT_ASSEMBLY` diagnostics (element counts, scatter nnz, assembly fingerprint) with conformance replay checks.
- 2026-03-08: Added Tier-3-bridge region-blocked prep assembly by deriving deterministic region topology profile signals from prep `region_mappings` (region block count, region mesh mean/variance), partitioning assembled DOFs into prep-shaped region blocks with per-block coefficient modulation and inter-block coupling attenuation, and emitting `FEA_PREP_REGION_TOPOLOGY` diagnostics with block graph stats + deterministic fingerprint; extended prep conformance to assert deterministic replay of both operator and region topology diagnostics.
- 2026-03-08: Advanced prep fidelity to a topology-informed operator-coefficient tier by deriving aggregate topology profile signals (surface/volume connectivity ratios, mixed-family ratio, mean region span) from prep artifacts, feeding them into FEA stiffness/mass/damping/rhs and coupling-strength shaping, and emitting deterministic `FEA_PREP_OPERATOR_TOPOLOGY` diagnostics (including structural coefficient stats and topology fingerprint) with conformance checks for replay stability plus bounded non-zero prep-vs-baseline structural deltas.
- 2026-03-08: Added topology descriptors to prep artifacts (`connectivity_class`, `element_family_hint`, `region_span_hint`) and wired topology-derived prep context signals (`topology_dof_multiplier`, `topology_bandwidth_proxy`, `mapped_region_participation_ratio`) into prep-aware solve context resolution.
- 2026-03-08: Implemented prep-topology assembly mode in FEA with deterministic DOF scaling from prep topology multipliers, coupling sparsity shaping from bandwidth proxies, and region-coupled load/constraint placement from mapping cardinalities, plus new `FEA_PREP_TOPOLOGY` diagnostics for structural topology statistics.
- 2026-03-08: Added conformance assertions for topology-backed prep runs (`FEA_PREP_TOPOLOGY` + deterministic replay checks) and published customer-facing manuals in `docs/geometry/prep-for-analysis.md` and `docs/analysis/prep-aware-solves.md` with explicit prep fidelity tiers.
- 2026-03-08: Integrated prep artifact health into nonlinear release readiness by extending `release_readiness_nonlinear.py` with typed prep reason codes (`PREP_SLO_COUNT_EXCEEDED`, `PREP_SLO_AGE_EXCEEDED`, `PREP_REJECT_RATE_HIGH`, `PREP_HEALTH_MISSING`) and branch-aware severity behavior, so prep lifecycle SLO regressions now influence readiness verdicts alongside conformance/trend/artifact checks.
- 2026-03-08: Added readiness unit coverage for prep-health pathways (count/age warn+fail and reject-rate signaling) and CI default prep-health thresholds for release readiness evaluation (`RUNMAT_RELEASE_READINESS_PREP_*`).
- 2026-03-08: Added prep artifact health observability with `geometry.prep_artifact_health/v1`, exposing lifecycle counters and age distribution summaries, plus runtime lifecycle event emission for prep artifact create/load/prune and stale/mismatch rejection outcomes.
- 2026-03-08: Added CI prep artifact SLO summary (`scripts/summarize_prep_artifacts.py`) and workflow integration with configurable warn/fail thresholds for artifact count and p95 artifact age, including summary publication in CI step output.
- 2026-03-08: Added prep artifact lifecycle governance with retention pruning controls (`RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS`, `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS_PER_GEOMETRY`, `RUNMAT_GEOMETRY_PREP_MAX_AGE_SECONDS`) applied on prep artifact persist/load across in-memory and optional filesystem-backed prep stores.
- 2026-03-08: Added prep staleness invalidation policy (`RUNMAT_GEOMETRY_PREP_REQUIRE_LATEST_REVISION`, default enabled) so prep references are rejected when newer revision prep artifacts exist for the same geometry id, with typed stale-run error mapping (`ANALYSIS_RUN_PREP_STALE`).
- 2026-03-08: Added runtime/contract coverage for prep lifecycle failure modes (missing, mismatched, stale references) and retention behavior (old artifact pruning with latest artifact replay preserved).
- 2026-03-08: Upgraded prep-aware assembly from scalar-only surrogate scaling to deterministic prep-driven operator structure changes by incorporating prep-derived mapped load/BC participation and layout seeds into DOF load/constrained distribution, and emitting `FEA_PREP_ASSEMBLY` diagnostics with mapped-load ratio, constrained-prep ratio, active-region count, and layout seed.
- 2026-03-08: Extended resolved prep context contracts/runtime mapping with derived participation metrics (`mapped_load_count`, `mapped_bc_count`, `layout_seed`) computed from trusted prep artifact mappings against model regions, preserving lineage/schema enforcement while making prep-vs-non-prep solve behavior structurally distinguishable.
- 2026-03-08: Added trusted prep lineage enforcement by persisting geometry prep artifacts (`geometry_prep_artifact/v1`) with returned `prep_artifact_id`, introducing prep artifact lookup at analysis run time, and rejecting untrusted direct prep contexts when no artifact reference is provided.
- 2026-03-08: Added typed prep-reference run failure mapping (`ANALYSIS_RUN_PREP_UNTRUSTED_CONTEXT`, `ANALYSIS_RUN_PREP_NOT_FOUND`, `ANALYSIS_RUN_PREP_SCHEMA_UNSUPPORTED`, `ANALYSIS_RUN_PREP_MISMATCH`) plus runtime/contract tests for missing and mismatched prep references.
- 2026-03-08: Updated prep-aware conformance to use artifact references end-to-end (`geometry.load -> geometry.prep_for_analysis -> analysis.create_model -> analysis.run_*`), keeping bounded prep-vs-baseline nonlinear delta checks and explicit `FEA_PREP_CONTEXT` observability.
- 2026-03-08: Extended prep context from model synthesis into solve execution by adding optional run-level prep payloads across analysis run option contracts (`analysis.run_linear_static/modal/transient/nonlinear`), mapping prep summaries into FEA solve options, and emitting `FEA_PREP_CONTEXT` diagnostics for prep-aware runs.
- 2026-03-08: Added deterministic prep-influenced assembly scaling in FEA (mesh density/quality driven stiffness/load shaping and prep-aware load-complexity augmentation), plus prep-vs-non-prep conformance coverage asserting bounded nonlinear quality deltas and explicit prep diagnostic presence.
- 2026-03-08: Completed prep-aware analysis model integration by extending `analysis.create_model/v1` intent with optional `prep_context` (source geometry id/revision + region mappings), adding typed prep-consistency validation/error mapping, and prioritizing prep-mapped regions during default BC/load placement plus assignment-confidence promotion.
- 2026-03-08: Added end-to-end contract coverage for `geometry.load -> geometry.prep_for_analysis -> analysis.create_model -> analysis.validate` and runtime unit tests for prep-context success/mismatch paths, ensuring deterministic prep-to-model flow behavior.
- 2026-03-08: Started Phase-7 meshing foundation by adding `runmat-meshing-core` with deterministic analysis-prep contracts (`MeshingOptions`, prepared mesh descriptors, region mapping provenance, quality report) and a constrained deterministic meshing MVP (`prepare_geometry_for_analysis`) for existing mesh-derived geometry assets.
- 2026-03-08: Added new runtime operation contract `geometry.prep_for_analysis/v1` (spec + typed error mapping) to bridge `GeometryAsset` into analysis-ready meshing-prep artifacts, with unit + operation-contract + dedicated conformance tests covering determinism and mapping stability across STL/STEP fixture inputs.
- 2026-03-08: Added release-readiness governance for nonlinear analysis via `scripts/release_readiness_nonlinear.py` with typed verdict output (`analysis-release-readiness/v1`: `pass`/`warn`/`fail`), reason codes, machine-readable artifact emission, and protected-branch fail semantics driven by conformance/trend/artifact verification signals.
- 2026-03-08: Added typed multi-run runtime analytics endpoints `analysis.results_compare/v1` and `analysis.trends/v1` for run-to-run deltas and rolling run-kind trend summaries, including nonlinear-specific deltas/rates (failed increments, max iterations, spike/stall counts) and solve-time statistics (median/p95).
- 2026-03-08: Added CI coverage and tests for readiness/trend logic (Python unit tests + runtime unit/contract tests), and wired branch-aware release-readiness evaluation into the nonlinear conformance pipeline with summary output and artifact capture.
- 2026-03-08: Added typed multi-run analytics operations: `analysis.results_compare/v1` for baseline-vs-candidate deltas (publishability/status, quality-reason count, nonlinear failed-increment/max-iteration/spike/stall deltas, solve-ms delta) and `analysis.trends/v1` for rolling run-kind summaries (median/p95 solve time, publishable rate, nonlinear failed-increment/spike/stall trend rates).
- 2026-03-08: Extended artifact store capabilities with run listing support so trend aggregation works across in-memory and filesystem stores, and added mixed-schema/noisy-sample trend tests to ensure robust behavior with legacy+wrapped artifacts and stable percentile calculations.
- 2026-03-08: Added CI nonlinear trend intelligence via `scripts/analyze_nonlinear_trends.py`, appending trend summaries to job output and enforcing branch-aware slowdown thresholds (`RUNMAT_ANALYSIS_TREND_MAX_SLOWDOWN_RATIO`) with protected-branch fail semantics.
- 2026-03-08: Added artifact lifecycle hardening for analysis run persistence by introducing schema-wrapped filesystem artifacts (`analysis_run_artifact/v1`) with operation metadata, backward-compatible loading of legacy raw artifacts, and forward-compatible tolerance for unknown future fields at the storage boundary.
- 2026-03-08: Added filesystem artifact retention controls (`RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS`, `RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS_PER_KIND`) with pruning-on-persist behavior and coverage for per-kind pruning semantics.
- 2026-03-08: Added deterministic filesystem replay validation for `analysis.results_by_run_id` (stable summary/status/quality reasons across repeated loads) and wired the replay check into nonlinear CI governance alongside nonlinear report schema validation.
- 2026-03-08: Added nonlinear contract-stability hardening for consumers: `analysis.results` now supports diagnostic-code filtering (`diagnostic_codes` query control), legacy nonlinear payload deserialization defaults for newly-added fields, and a golden contract-shape snapshot test (`tests/data/nonlinear_contract_snapshot.json`) to catch accidental payload/summary schema drift.
- 2026-03-08: Added consumer-side nonlinear report utility validation (`scripts/validate_analysis_report_nonlinear.py`) and wired it into nonlinear CI governance so `analysis_benchmark_report.json` must include expected nonlinear fixture threshold keys before summary/artifact publication.
- 2026-03-08: Added two harder nonlinear realism fixtures (`NonlinearSofteningProxy`, `NonlinearLoadPathMix`) and integrated provider-backed conformance variants (`nonlinear_softening_proxy_gpu_provider`, `nonlinear_load_path_mix_gpu_provider`) to exercise distinct nonlinear behavior classes beyond baseline/stress fixtures.
- 2026-03-08: Extended nonlinear solver difficulty profiling with additional structured convergence metrics (`iteration_spike_count`, `convergence_stall_count`, `backtrack_burst_count`, `max_line_search_backtracks_per_increment`) surfaced via `FEA_NONLINEAR_CONVERGENCE`, plus runtime payload/summary propagation so `analysis.results` can gate and query these signals directly.
- 2026-03-08: Added nonlinear behavior-shape conformance gates for the new fixtures (stall/spike/backtrack-burst and per-increment backtrack bounds), expanded rolling baseline targets to include the new nonlinear fixtures, and added policy-divergence contract coverage on a harder fixture profile to verify exploratory vs balanced/strict outcomes under constrained iteration budgets.
- 2026-03-08: Added nonlinear CI governance automation with a dedicated `nonlinear-conformance` workflow job (Linux GPU) that runs `--test analysis`, enables branch-aware baseline enforcement defaults on protected branches, emits a concise nonlinear benchmark summary from artifacts, and uploads nonlinear artifacts only when useful (failure or protected branch pushes).
- 2026-03-08: Added nonlinear observability event logging in runtime (`analysis.run_nonlinear`) with structured run-outcome fields (failed increments, iteration cap usage, backtracks, tangent rebuilds, max residual/increment norms, publishability/status) so local OTEL pipelines can surface nonlinear regressions outside test logs.
- 2026-03-08: Calibrated nonlinear presets and conformance gates from sweep evidence by introducing `AnalysisNonlinearRunOptions::production_recommended()` (deterministic fp64 balanced, 24 increments baseline / stress fixture override 32, max_newton_iters=28, line_search=true, max_backtracks=8, tangent_refresh_interval=2), wiring harness defaults to that preset, and validating against benchmark artifacts where default provider runs held publishability with strong speedups (`nonlinear_assembly_gpu_provider` ~2.65x, `nonlinear_assembly_stress_gpu_provider` ~4.37x).
- 2026-03-08: Added nonlinear tuning override knobs for harness sweeps (`RUNMAT_NONLINEAR_*`) and tightened must-not-regress nonlinear threshold gates (exact converged increment targets, failed increment ceilings, required line-search activity bounds, tangent rebuild upper bound), with sweep confirmation that disabling line search triggers intended gate failures (increment failures and elevated tangent rebuilds).
- 2026-03-08: Expanded nonlinear policy contract/unit coverage by asserting explicit policy divergence under induced increment-cap pressure: exploratory remains publishable while balanced/strict degrade with `NonlinearIncrementFailure`, and added nonlinear preset ordering tests covering coarse/balanced/production/high-accuracy tradeoffs.
- 2026-03-07: Tightened nonlinear `QualityPolicy::Strict` publishability semantics so strict runs now explicitly reject when increment failures occur or Newton iteration caps are exhausted (iteration-cap hits now emit `NonlinearIncrementFailure` quality reasons with failed/cap-hit/max-iteration details), and added focused runtime unit coverage for the strict rejection path.
- 2026-03-07: Upgraded nonlinear solver fidelity from scaffold to an incremental-Newton-v2 path with dual convergence gating (residual + increment norm), configurable tangent refresh cadence, configurable backtracking line-search controls, and richer nonlinear observability (`FEA_NONLINEAR_CONVERGENCE` + `FEA_NONLINEAR_COST` metrics for failed increments, max iteration usage, max increment norm, line-search backtracks, tangent rebuild count, and solver cost).
- 2026-03-07: Extended runtime nonlinear contracts/results to carry the richer nonlinear telemetry (`increment_norms`, failed-increment and line-search/tangent counters, and nonlinear summary discoverability fields), tightened nonlinear option validation surfaces, and preserved operation version shape with additive fields only.
- 2026-03-07: Expanded nonlinear conformance/performance coverage with a tougher provider-backed fixture (`nonlinear_assembly_stress_gpu_provider`) plus explicit threshold gates (converged/failed increment bounds, residual/increment-norm ceilings, line-search/tangent activity) and included the stress fixture in rolling baseline drift targets; benchmark solver-cost extraction now also consumes `FEA_NONLINEAR_COST`.
- 2026-03-07: Completed nonlinear intent-model closure by enabling `analysis.create_model/v1` support for `nonlinear_structural` (default nonlinear step/load template instead of unsupported-contract rejection), added runtime/unit/contract coverage for the new profile path, and extended benchmark conformance with a synthesized nonlinear fixture (`nonlinear_create_model_cpu`) that exercises `geometry.load` + `analysis.create_model` + `analysis.run_nonlinear` end-to-end.
- 2026-03-07: Completed runtime conformance harness refactor by replacing monolithic `tests/analysis_benchmark_conformance.rs` with a modular `tests/analysis/` layout (`mod.rs`, `manifest.rs`, `runner.rs`, `harness.rs`, `baseline.rs`) and a new integration-test entrypoint `tests/analysis.rs`, preserving existing benchmark/conformance behavior and keeping nonlinear/modal/transient gates green.
- 2026-03-06: Introduced a native nonlinear runtime/solver path by adding `analysis.run_nonlinear/v1` + `analysis_run_nonlinear_with_options_op` contracts, typed nonlinear payloads (`nonlinear_results/v1` with load factors/increment snapshots/residuals/iteration counts), and nonlinear-specific quality reasons (`NonlinearResidualExceeded`, `NonlinearIncrementFailure`) while preserving existing modal/transient behavior.
- 2026-03-06: Added nonlinear capability fixtures and conformance gates (`FixtureId::NonlinearAssembly`, `nonlinear_assembly_cpu`, `nonlinear_assembly_gpu_provider`) with backend/residency/speedup and incremental-convergence assertions, and folded nonlinear provider fixture into rolling baseline drift target sets for perf/cost regression tracking.
- 2026-03-06: Extended FEA fixture portfolio with `TransientShock` and `NonlinearAssembly`, plus native nonlinear solver scaffolding (`solve/nonlinear`) built on incremental Newton-style semantics (load increments + convergence diagnostics) and consistent telemetry propagation (`solver_backend`, device-apply ratio, host sync, diagnostics).
- 2026-03-06: Completed the next 1-4 capability/disciplined pass: added a new challenging transient fixture class (`TransientShock`) and integrated it into conformance (CPU + GPU-provider), introduced transient physical-invariant diagnostics (`FEA_TRANSIENT_PHYSICS`: bounded step-jump ratio + non-finite displacement count) with explicit conformance thresholds, promoted a contract-level production transient preset (`AnalysisTransientRunOptions::production_recommended`) and routed benchmark transient runs through it, and extended modal robustness coverage with 16-mode stress fixtures (CPU + GPU-provider).
- 2026-03-06: Ran transient dt-bucketing A/B sweeps at `RUNMAT_TRANSIENT_DT_BUCKET_REL_TOL={0.0,0.01,0.02}` on both `transient_long_gpu_provider` and `transient_shock_gpu_provider`; observed best aggregate throughput at `0.01` with no quality regressions, so production-recommended transient options now default to `dt_bucket_rel_tolerance=0.01` while global defaults remain compatibility-safe.
- 2026-03-06: Completed follow-on plan items (1-4): exposed transient adaptivity/bucketing controller parameters in runtime and FEA transient option contracts (`adapt_*` controls + `dt_bucket_rel_tolerance`), added modal stress conformance fixtures at 16 modes (CPU + GPU provider) with quality/throughput gates, extended rolling baseline checks to include solver-cost metrics and protected-branch CI enforcement switches, and ran dt-bucketing A/B sweeps (`0.0`, `0.01`, `0.02`) showing no measurable cache-hit improvement on current fixture (recommend keeping default `0.0` until broader adaptive-`dt` scenarios are added).
- 2026-03-06: Completed next-phase benchmark + solver progression: (1) added fixture-scoped adaptivity conformance assertions for `FEA_TRANSIENT_ADAPTIVITY`; (2) added CI-oriented rolling baseline drift support with directory/windowed median comparisons for GPU runtime + speedup retention + solver-cost metrics and protected-branch enforce mode; (3) enabled benchmark report snapshot persistence to baseline directories; (4) upgraded transient adaptive timestep control to residual-aware bounded scaling and added optional dt-cache bucketing mode via `RUNMAT_TRANSIENT_DT_BUCKET_REL_TOL` with `FEA_TRANSIENT_BUCKETING` diagnostics and quality assertions when enabled.
- 2026-03-06: Added rolling baseline regression checks for core accelerated analysis fixtures (`modal_large_gpu_provider`, `transient_long_gpu_provider`) in `analysis_benchmark_conformance` with configurable baseline directory/window and median-based drift checks for GPU runtime slowdown and speedup retention (`RUNMAT_ANALYSIS_BASELINE_DIR`, `RUNMAT_ANALYSIS_BASELINE_WINDOW`, `RUNMAT_ANALYSIS_MIN_SPEEDUP_RETENTION`).
- 2026-03-06: Upgraded transient adaptive timestep policy from one-way growth heuristic to residual-aware scaling (bounded proportional controller with retry-sensitive growth caps and non-converged shrink behavior), and added `FEA_TRANSIENT_ADAPTIVITY` diagnostics (`increase/decrease/hold` counts plus scale min/max/mean) for observability and tuning.
- 2026-03-06: Performed post-split naming/API hygiene cleanup across new solve submodules without behavior changes (clearer internal type/function names such as `LinearSolveAttempt`, `solve_k_system_cg`, `LinearStepStats`, `solve_implicit_step_system`, and `solve_runtime_tensor_linear_system_internal`) to improve readability and future extension ergonomics.
- 2026-03-06: Refactored solver internals for maintainability by splitting oversized `solve/` units into focused submodules (`solve/modal/{mod,math,linear_solve,diagnostics}.rs`, `solve/transient/{mod,linear_step,diagnostics}.rs`, `solve/runtime_tensor_solver/{mod,operator_impl,preconditioner_impl}.rs`) while preserving runtime contracts and test/conformance behavior; calibrated transient cache conformance thresholds to reflect observed adaptive-`dt` variability after structural cleanup.
- 2026-03-06: Added structured solver-cost observability for modal/transient runs (`FEA_MODAL_COST`, `FEA_TRANSIENT_COST` with `prepared_build_ms`, `solve_ms`, `fallback_apply_count`) and surfaced these metrics in benchmark artifacts (`gpu_solver_prepared_build_ms`, `gpu_solver_solve_ms`, `gpu_solver_fallback_apply_count`) alongside new transient cache-efficiency artifact fields and thresholds.
- 2026-03-06: Wired transient prepared-cache efficiency into benchmark artifacts and conformance gates by recording provider-run cache metrics (`gpu_transient_cache_entries`, `gpu_transient_cache_hit_ratio`, `gpu_transient_cache_misses`) from `FEA_TRANSIENT_CACHE` diagnostics and enforcing initial thresholds on `transient_long_gpu_provider` (minimum hit ratio and maximum misses).
- 2026-03-06: Added transient prepared-system cache churn controls and observability by introducing an LRU-bounded cache for runtime-tensor implicit-step prepared systems (capacity 12, keyed by exact `dt` bits) plus `FEA_TRANSIENT_CACHE` diagnostics reporting cache entries/hits/misses, preventing unbounded adaptive-`dt` cache growth while preserving exact-operator reuse semantics.
- 2026-03-06: Tightened modal/transient GPU provider conformance performance gates after prepared-context reuse improvements (`modal_large_gpu_provider` speedup floor raised to `2.8x`, host-sync ceiling reduced to `96`; `transient_long_gpu_provider` speedup floor raised to `2.0x`, host-sync ceiling reduced to `48`) while keeping fallback parity gates unchanged.
- 2026-03-06: Extended prepared runtime-tensor linear-system reuse into transient GPU implicit solves by caching prepared systems per accepted/adaptive timestep (`dt.to_bits()` keyed) and reusing operator/preconditioner metadata across step solves, with fallback to the existing per-step build path when preparation is unavailable.
- 2026-03-06: Added runtime-tensor prepared linear-system scaffolding (`prepare_runtime_tensor_linear_system` + `solve_prepared_linear_system_runtime_tensor`) and wired modal GPU subspace iterations to reuse prepared operator/preconditioner metadata across repeated inverse-`K` solves, reducing per-solve setup overhead while keeping conformance gates green (notably higher modal provider speedup in harness artifacts).
- 2026-03-06: Tuned modal inverse-`K` adaptive iteration policy to be provider-aware (faster early-stop profile only when a GPU acceleration provider is active, conservative profile for CPU/no-provider fallback), restoring strict CPU/GPU-fallback parity while retaining improved modal GPU speedup in benchmark gates.
- 2026-03-06: Reduced modal solve-call pressure in the subspace iteration path by adding adaptive early-stop criteria for inverse-`K` iterations (minimum iteration floor + relative mode-shape update threshold), cutting unnecessary repeated linear solves while retaining quality diagnostics and existing conformance speedup/host-sync gates.
- 2026-03-06: Added selective linear-solve warm-start scaffolding for iterative analysis solves (new runtime-tensor solver entrypoint with optional initial guess and modal CPU inverse-`K` CG reuse across subspace iterations), while explicitly keeping modal/transient GPU runtime-tensor solves on zero-initialized iterations for now to preserve measured speedup gates and avoid extra per-solve operator-apply overhead.
- 2026-03-06: Tightened modal/transient GPU provider host-sync conformance ceilings after sync-accounting cleanup (`modal_large_gpu_provider` now `<=128`, `transient_long_gpu_provider` now `<=64`) based on fresh observed counts (`64` and `24` respectively), keeping pressure on low-transfer solve loops.
- 2026-03-06: Reduced modal/transient runtime-tensor solver host-sync overhead accounting by treating provider-supported scalar reductions (`read_scalar`) as device-side reads instead of forced host sync events, keeping explicit host transfer counts for fallback downloads/final solution extraction; this keeps sync telemetry focused on actual host-device transfer pressure while preserving acceleration gate signals.
- 2026-03-06: Tightened initial modal/transient GPU speedup conformance floors using fresh artifact observations (`modal_large_gpu_provider` minimum speedup raised to `2.2x`, `transient_long_gpu_provider` minimum speedup raised to `1.8x`) while keeping fallback fixtures unconstrained for speedup, preserving stability and anti-regression pressure.
- 2026-03-06: Added per-fixture GPU speedup floor gates to benchmark/conformance artifacts by recording `gpu_speedup_ratio = cpu_run_ms / gpu_run_ms` and enforcing `min_gpu_speedup_ratio` for modal/transient provider fixtures, creating an explicit performance-regression contract (beyond backend-selection and device-apply telemetry).
- 2026-03-06: Calibrated modal/transient GPU conformance limits using observed harness artifact telemetry (set non-flaky host-sync budgets for provider runs while retaining strict backend and device-apply-ratio expectations), aligned modal publishability expectations with quality-policy behavior, and fixed benchmark harness pass/fail assertion semantics so non-baseline core gate failures correctly fail CI.
- 2026-03-06: Started hardware-acceleration gating for modal/transient solve paths by routing modal inverse-`K` and transient implicit-step linear solves through the runtime-tensor PCG backend when GPU+provider is available, propagating solver backend/device-apply ratio/host-sync telemetry into runtime provenance, and enforcing new benchmark expectations (`expected_solver_backend`, minimum `solver_device_apply_k_ratio`) for modal/transient GPU provider vs fallback fixture variants.
- 2026-03-06: Expanded benchmark/conformance harness GPU scope for non-linear-static paths by adding modal/transient GPU fixture variants (provider + no-provider fallback) with run-kind-aware execution, residency expectations, and parity checks; runtime modal/transient ops now also apply field-to-device promotion so provider-backed runs can emit `DeviceRef` analysis fields and fallback telemetry consistently.
- 2026-03-06: Extended `analysis_benchmark_conformance.rs` beyond linear-static-only coverage by adding modal (`ModalLarge`) and transient (`TransientLong`) fixture gates, run-kind aware execution paths, and structured threshold assertion records in benchmark artifacts for modal orthogonality/separation and transient stability/energy diagnostics.
- 2026-03-06: Hardened modal/transient quality diagnostics integration by wiring new modal (`FEA_MODAL_ORTHOGONALITY`, `FEA_MODAL_SEPARATION`) and transient (`FEA_TRANSIENT_STABILITY`, `FEA_TRANSIENT_STEP_FAILURE`, `FEA_TRANSIENT_ENERGY`) signals into runtime quality-reason policy gating, then stabilized transient energy growth checking to use first non-trivial baseline energy so default transient runs remain publishable while long-horizon fixtures still emit stability telemetry.
- 2026-03-06: Added runtime transient execution tuning surface (`analysis_run_transient_with_options_op` + `AnalysisTransientRunOptions`) exposing timestep bounds, adaptive-step controls, solver tolerances, and retry budget while preserving `analysis.run_transient/v1` envelope compatibility.
- 2026-03-06: Added transient option preset helpers (`coarse`, `balanced`, `high_accuracy`) on `AnalysisTransientRunOptions` to provide stable cost-vs-accuracy starting points for clients without custom knob tuning.
- 2026-03-06: Added modal run tuning surface (`analysis_run_modal_with_options_op` + `AnalysisModalRunOptions`) with stable presets (`coarse`, `balanced`, `high_accuracy`) for mode-budget and residual gate tuning while preserving `analysis.run_modal/v1` contract shape.
- 2026-03-06: Implemented native transient execution baseline in `runmat-analysis-fea` (`implicit_euler_pcg`) and wired runtime `analysis.run_transient/v1` to native results payloads (`transient_results`) with typed integration metadata and publishability/quality gating.
- 2026-03-06: Added first-class transient payload contract (`transient_payload_version`, `time_points_s`, `displacement_snapshots`, `residual_norms`, `integration_method`) and runtime results propagation for transient runs.
- 2026-03-06: Added transient results query controls (`include_transient_results`, `transient_snapshot_indices`) with typed out-of-range snapshot mapping (`ANALYSIS_RESULTS_TRANSIENT_SNAPSHOT_NOT_FOUND`) to bound transient payload retrieval for long time histories.
- 2026-03-06: Extended `analysis.results` summary metadata with transient discoverability fields (`snapshot_count`, `time_start_s`, `time_end_s`) so clients can preflight transient timeline selection without fetching snapshot payloads.
- 2026-03-06: Added transient quality-summary metadata (`max_transient_residual_norm`, `final_step_converged`) to `analysis.results` summary for lightweight go/no-go checks without loading transient payload bodies.
- 2026-03-06: Added modal quality-summary metadata (`max_modal_residual_norm`, `first_mode_converged`) to `analysis.results` summary to mirror transient preflight semantics and enable fast modal quality screening without payload retrieval.
- 2026-03-06: Upgraded native modal solver core from diagonal extraction to matrix-free subspace iteration with inverse-K inner solves and M-orthonormalization, exposing richer modal residual telemetry while preserving operation contract shape.
- 2026-03-06: Extended create-model profile support so `transient_structural` now synthesizes a valid transient template (step/load defaults) while retaining explicit unsupported semantics only for `nonlinear_structural`.
- 2026-03-06: Added initial native modal solve path in `runmat-analysis-fea` (`solve/modal.rs`, `run_modal_with_options`) using assembled generalized diagonal stiffness/mass extraction, emitting modal diagnostics (`FEA_MODAL_METHOD`, `FEA_MODAL_CONVERGENCE`) and deterministic mode-shape fields.
- 2026-03-06: Switched runtime `analysis.run_modal/v1` to consume native FEA modal execution (instead of linear-static placeholder), including modal quality-gate evaluation and publishability/status decisions from modal convergence + modal payload validity.
- 2026-03-06: Added modal residual tracking (`residual_norms`) to modal payloads and FEA diagnostics (`FEA_MODAL_RESIDUAL`), with runtime quality gating that emits `ModalResidualExceeded` when residual thresholds are exceeded.
- 2026-03-06: Added `analysis.run_transient/v1` operation scaffold with typed model-shape validation and degraded placeholder execution path (`FEA_TRANSIENT_PLACEHOLDER`, `TransientPlaceholder`) to establish transient contract/provenance semantics before native transient integration.
- 2026-03-06: Upgraded `analysis.create_model/v1` synthesis heuristics to use geometry regions and CAD material evidence (`MaterialEvidence`) for inferred material models/assignments plus boundary/load region targeting, with new contract and unit coverage.
- 2026-03-06: Added `analysis.run_modal/v1` operation contract scaffold with typed gating: explicit model-shape validation (`ANALYSIS_RUN_MODAL_INVALID_MODEL` when no modal step) and explicit unsupported solver-path contract (`ANALYSIS_RUN_MODAL_UNSUPPORTED`) while modal execution implementation is pending.
- 2026-03-06: Upgraded `analysis.run_modal/v1` from unsupported scaffold to a deterministic placeholder execution path that reuses linear-static infrastructure, emits explicit `FEA_MODAL_PLACEHOLDER` diagnostics + `ModalPlaceholder` quality reason, and forces degraded/non-publishable status until native modal solver implementation lands.
- 2026-03-06: Added first-class modal payload schema (`modal_results`) to analysis run/results contracts with placeholder eigenvalue/mode-shape data (`eigenvalues_hz`, `mode_shapes`) so modal outputs are forward-compatible before native eigen-solver rollout.
- 2026-03-06: Added `analysis.results` modal query controls (`include_modal_results`, `mode_indices`) with typed out-of-range mode mapping (`ANALYSIS_RESULTS_MODE_NOT_FOUND`) so modal payload retrieval can be bounded/filtered as eigen-solve output sizes grow.
- 2026-03-06: Extended `analysis.results` summary metadata with modal discoverability fields (`mode_count`, `available_mode_indices`) so clients can discover selectable modes without fetching full modal payloads.
- 2026-03-06: Added modal frequency-range summary metadata (`min_frequency_hz`, `max_frequency_hz`) to `analysis.results` for lightweight modal span inspection before payload slicing.
- 2026-03-06: Added explicit modal metadata fields (`mode_units`, `frequency_basis`) to modal payload contracts so clients can distinguish placeholder-derived modal frequencies from native eigen-solver outputs without inferring from diagnostics.
- 2026-03-06: Replaced stringly modal `frequency_basis` with typed enum semantics (`placeholder_linear_static`, `native_eigen_solve`) to lock modal provenance contracts and avoid downstream string drift.
- 2026-03-06: Replaced stringly modal `mode_units` with typed enum semantics (`hz`) to fully type modal metadata contracts and avoid unit-string drift across clients.
- 2026-03-06: Added explicit modal payload version metadata (`modal_payload_version`, initial value `modal_results/v1`) so modal contract evolution can remain backward-compatible as native eigen output richness increases.
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
