# Code-Reviewed FEA Workflow (Study Document Model)

## Purpose

Define a readable, composable, code-reviewed FEA workflow for RunMat where:

- the primary setup artifact is a single study document (`study.yaml` by default),
- `.m` scripts stay lightweight and call high-level study builtins,
- runtime resolves study data into typed operation execution,
- outputs remain machine-readable evidence artifacts for automation/governance,
- shared reusable setup blocks (materials, loads, policies, templates) are includable.

## Design Direction

Use one authored input model:

- **Authored source of truth**: `study.yaml` (JSON also accepted).
- **Execution API**: typed builtins (`analysis.validate_study`, `analysis.plan_study`, `analysis.run_study`).
- **Evidence outputs**: JSON artifacts plus a native RunMat dataset under root artifact storage (`.artifacts/analysis/by-id/<analysis_id>/runs/<run_id>/run.data`).

This keeps authoring readable while preserving strict typed runtime behavior.

## Why This Is Better

- readable reviews (engineers inspect one study doc + tiny `.m`),
- no dual control plane (workflow logic is not duplicated across `.m` and JSON),
- composability and reuse via includes/imports,
- natural fit for `serde` schema validation and typed operation contracts,
- straightforward path to model-agent generation and engineer review.

## Canonical v1 Package Layout

```text
simulations/
  bracket-nonlinear-v1/
    study.yaml
    scripts/
      run_simulation.m
      inspect_results.m
    includes/
      materials/
        steel_library.yaml
      loads/
        pressure_cases.yaml
      policies/
        release_balanced.yaml

.artifacts/
  analysis/
    by-id/
      ana_01HXYZ.../
        meta.json
        latest.json
        runs/
          run_01HXYZ.../
            study_lock.json
            run_manifest.json
            analysis_results.json
            analysis_diagnostics.json
            analysis_trends.json
            readiness_verdict.json
            run.data/
              manifest.json
              arrays/
    by-study/
      sha256_3f.../
        pointer.json
    index/
      analyses.jsonl
      runs.jsonl
```

Notes:

- `study.yaml` is the primary reviewed setup artifact.
- `includes/` contains shared fragments reused across studies.
- `<workspace-root>/.artifacts/analysis/` is the canonical lineage root.
- `.artifacts/analysis/by-id/<analysis_id>/runs/<run_id>/` is the immutable execution root.
- `run.data` under each run root is the canonical script-facing inspection surface.

## Builtin Surface (Planned)

Status:

- Planned study builtins: `analysis.validate_study`, `analysis.plan_study`, `analysis.run_study`.
- Planned discovery helpers: `analysis.resolve_study`, `analysis.latest_run`, `analysis.open_run`, `analysis.open_run_data`.
- Interim behavior: callers may read index/pointer files directly using the contracts in this document.

- `analysis.validate_study(path, options?)`
  - parses YAML/JSON,
  - resolves includes,
  - validates typed schema and references,
  - returns typed diagnostics/errors.

- `analysis.plan_study(path, options?)`
  - returns resolved execution plan without running solve,
  - useful for review tooling and CI preflight.

- `analysis.run_study(path, options?)`
  - executes the full canonical sequence internally,
  - emits standard evidence artifacts and run metadata under `.artifacts/analysis/by-id/<analysis_id>/runs/<run_id>/`.

Implementation guidance:

- register these through `#runmat_builtin`,
- back with `serde` structs/enums and registry metadata,
- keep additive schema evolution (`runmat-analysis-study/v1`, `v2`, ...).

## Study Schema Sketch (`runmat-analysis-study/v1`)

Whole-schema example aligned to currently implemented runtime/core interfaces:

```yaml
schema: runmat-analysis-study/v1
study_id: bracket_nonlinear_v1
revision: 7

imports:
  - path: includes/materials/steel_library.yaml
  - path: includes/loads/pressure_cases.yaml
  - path: includes/policies/release_balanced.yaml

geometry:
  source_path: assets/bracket.step
  source_kind: step
  expected_units: millimeter

create_model_intent:
  model_id: bracket_model_v1
  profile: nonlinear_structural
  # Optional; shape matches AnalysisCreateModelPrepContext
  prep_context:
    source_geometry_id: geo_bracket
    source_geometry_revision: 3
    region_mappings:
      - region_id: bracket_body
        source_mesh_ids: [mesh_body]
        prepared_mesh_ids: [prep_mesh_body]
      - region_id: mount_holes
        source_mesh_ids: [mesh_holes]
        prepared_mesh_ids: [prep_mesh_holes]

model:
  frame: global
  units: millimeter
  materials:
    - material_id: steel_1018
      name: AISI 1018
      mechanical:
        youngs_modulus_pa: 2.1e11
        poisson_ratio: 0.29
      thermal:
        reference_temperature_k: 293.15
        modulus_temp_coeff_per_k: -2.5e-4
        conductivity_w_per_mk: 45.0
        specific_heat_j_per_kgk: 500.0
        expansion_coefficient_per_k: 1.2e-5
      electrical:
        reference_temperature_k: 293.15
        conductivity_s_per_m: 1.0
        resistive_heating_coefficient: 0.0
      plastic:
        yield_strain: 0.002
        hardening_modulus_ratio: 0.02
        saturation_exponent: 2.0
  material_assignments:
    - region_id: bracket_body
      expected_material_id: steel_1018
      assigned_material_id: steel_1018
      confidence: verified
    - region_id: mount_holes
      expected_material_id: steel_1018
      assigned_material_id: steel_1018
      confidence: probable
  boundary_conditions:
    - bc_id: bc_mount_fixed
      region_id: mount_holes
      kind: fixed
  loads:
    - load_id: load_service_pressure
      region_id: top_face
      kind:
        pressure:
          magnitude_pa: 2500000.0
    - load_id: load_gravity
      region_id: bracket_body
      kind:
        body_force:
          gx: 0.0
          gy: -9.81
          gz: 0.0
  steps:
    - step_id: step_nonlinear
      kind: nonlinear
  interfaces:
    - interface_id: contact_mount
      primary_region_id: bracket_body
      secondary_region_id: mount_holes
      kind:
        contact:
          penalty_stiffness_scale: 1.0
          max_penetration_ratio: 0.01
          friction_coefficient: 0.0

execution:
  backend: gpu
  options:
    deterministic_mode: true
    precision_mode: fp64
    quality_policy: balanced
    increment_count: 16
    max_newton_iters: 30
    tolerance: 1.0e-6
    residual_convergence_factor: 4.0
    increment_norm_tolerance: 1.0e-7
    line_search: true
    max_line_search_backtracks: 6
    line_search_reduction: 0.5
    tangent_refresh_interval: 2
    prep_artifact_id: prep_bracket_v3
    prep_calibration_profile: balanced

results_query:
  include_fields: []
  include_diagnostics: true
  diagnostic_codes: []
  include_modal_results: true
  mode_indices: []
  include_transient_results: true
  transient_snapshot_indices: []
  include_nonlinear_results: true
```

JSON equivalent is accepted for automation-heavy contexts.

## Schema Fields (`runmat-analysis-study/v1`)

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| `schema` | yes | string | Must equal `runmat-analysis-study/v1`. |
| `study_id` | yes | string | Stable logical id for review lineage. |
| `revision` | yes | integer | Monotonic package revision for auditability. |
| `imports` | no | list of import objects | Ordered include list; resolved before validation/execution. |
| `geometry` | yes | object | Maps to geometry load input (`source_path`, optional `source_kind`, `expected_units`). |
| `create_model_intent` | yes | object | Maps to `AnalysisCreateModelIntentSpec` (`model_id`, `profile`, optional `prep_context`). |
| `model` | yes | object | Maps to `AnalysisModel` fields (`frame`, `units`, `materials`, `material_assignments`, `boundary_conditions`, `loads`, `steps`). |
| `execution` | yes | object | Contains `backend` (`cpu` or `gpu`) and profile-matched run options. |
| `results_query` | no | object | Maps to `AnalysisResultsQuery`; defaults to runtime defaults when omitted. |
| `quality_policy` | no | object | Optional higher-level package policy metadata; run option policy still comes from `execution.options`. |
| `acceptance` | no | object | Acceptance profile/threshold mapping (often required in CI profiles). |
| `metadata` | no | object | Ownership/tags/notes/ticket links; ignored by solver path. |

## Rust Type Mapping (Normative v1)

Study field to currently implemented type mapping:

| Study path | Required | Source | Rust type |
| --- | --- | --- | --- |
| `create_model_intent` | yes | study | `runmat_runtime::analysis::contracts::AnalysisCreateModelIntentSpec` |
| `create_model_intent.profile` | yes | study | `runmat_runtime::analysis::contracts::AnalysisCreateModelProfile` |
| `create_model_intent.prep_context` | no | study | `runmat_runtime::analysis::contracts::AnalysisCreateModelPrepContext` |
| `create_model_intent.prep_context.region_mappings[]` | conditional | study | `runmat_meshing_core::RegionMeshMapping` |
| `model` | yes | study | `runmat_analysis_core::problem::model::AnalysisModel` (study-owned fields only; `model_id`, `geometry_id`, `geometry_revision` are resolved during execution) |
| `model.frame` | yes | study | `runmat_analysis_core::problem::model::ReferenceFrame` |
| `model.units` | yes | study | `runmat_geometry_core::UnitSystem` |
| `model.materials[]` | conditional | study | `runmat_analysis_core::problem::materials::MaterialModel` |
| `model.materials[].mechanical` | conditional | study | `runmat_analysis_core::problem::materials::MaterialMechanicalModel` |
| `model.materials[].thermal` | conditional | study | `runmat_analysis_core::problem::materials::MaterialThermalModel` |
| `model.materials[].electrical` | conditional | study | `runmat_analysis_core::problem::materials::MaterialElectricalModel` |
| `model.materials[].plastic` | conditional | study | `runmat_analysis_core::problem::materials::MaterialPlasticModel` |
| `model.material_assignments[]` | yes | study | `runmat_analysis_core::problem::material_assignment::MaterialAssignment` |
| `model.material_assignments[].confidence` | no | study | `runmat_analysis_core::problem::material_assignment::EvidenceConfidence` |
| `model.boundary_conditions[]` | yes | study | `runmat_analysis_core::problem::bc::BoundaryCondition` |
| `model.boundary_conditions[].kind` | yes | study | `runmat_analysis_core::problem::bc::BoundaryConditionKind` |
| `model.loads[]` | yes | study | `runmat_analysis_core::problem::loads::LoadCase` |
| `model.loads[].kind` | yes | study | `runmat_analysis_core::problem::loads::LoadKind` |
| `model.steps[]` | yes | study | `runmat_analysis_core::problem::steps::AnalysisStep` |
| `model.steps[].kind` | yes | study | `runmat_analysis_core::problem::steps::AnalysisStepKind` |
| `model.interfaces[]` | conditional | study | `runmat_analysis_core::problem::interfaces::AnalysisInterface` |
| `model.interfaces[].kind` | conditional | study | `runmat_analysis_core::problem::interfaces::AnalysisInterfaceKind` |
| `execution.backend` | yes | study | `runmat_analysis_fea::ComputeBackend` |
| `execution.options` (linear static) | conditional | study | `runmat_runtime::analysis::contracts::AnalysisRunOptions` |
| `execution.options` (modal) | conditional | study | `runmat_runtime::analysis::contracts::AnalysisModalRunOptions` |
| `execution.options` (thermal) | conditional | study | `runmat_runtime::analysis::contracts::AnalysisThermalRunOptions` |
| `execution.options` (transient) | conditional | study | `runmat_runtime::analysis::contracts::AnalysisTransientRunOptions` |
| `execution.options` (nonlinear) | conditional | study | `runmat_runtime::analysis::contracts::AnalysisNonlinearRunOptions` |
| `execution.options.precision_mode` | no | study/defaulted | `runmat_runtime::analysis::contracts::PrecisionMode` |
| `execution.options.quality_policy` | no | study/defaulted | `runmat_runtime::analysis::contracts::QualityPolicy` |
| `execution.options.prep_calibration_profile` | no | study | `runmat_runtime::analysis::contracts::PrepCalibrationProfile` |
| `results_query` | no | study/defaulted | `runmat_runtime::analysis::contracts::AnalysisResultsQuery` |

Runtime operation bindings used by study builtins:

| Builtin stage | Required | Runtime function |
| --- | --- | --- |
| create model | yes | `analysis_create_model_op` |
| run linear static | conditional (`profile=linear_static_structural`) | `analysis_run_linear_static_with_options` |
| run modal | conditional (`profile=modal_structural`) | `analysis_run_modal_with_options_op` |
| run thermal | conditional (`profile=thermal_standalone`) | `analysis_run_thermal_with_options_op` |
| run transient | conditional (`profile=transient_structural`) | `analysis_run_transient_with_options_op` |
| run nonlinear | conditional (`profile=nonlinear_structural`) | `analysis_run_nonlinear_with_options_op` |

Import object fields:

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| `path` | yes | string | Relative path under study root unless explicit allowlist says otherwise. |
| `alias` | no | string | Optional namespace label for diagnostics/readability. |
| `required` | no | bool | Defaults `true`; when `false`, missing import emits warning not hard failure. |

Execution option shapes by profile (matching current runtime structs):

- linear static: `AnalysisRunOptions`.
- modal: `AnalysisModalRunOptions`.
- thermal: `AnalysisThermalRunOptions` (`step_count`, `time_step_s`, `residual_warn_threshold`).
- transient: `AnalysisTransientRunOptions`.
- nonlinear: `AnalysisNonlinearRunOptions` (`increment_count`, `max_newton_iters`, `tolerance`, `residual_convergence_factor`, `increment_norm_tolerance`, `line_search`, `max_line_search_backtracks`, `line_search_reduction`, `tangent_refresh_interval`).

## Companion Schemas

- `runmat-analysis-study-lock/v1`
  - fully resolved study document after import merge,
  - hashes for root study + imported fragments,
  - resolved profile/run options,
  - builtin/runtime/schema version metadata.

- `runmat-analysis-plan/v1`
  - output from `analysis.plan_study`,
  - selected run op and operation sequence,
  - derived checks and warnings before solve.

- `runmat-analysis-study-validation/v1`
  - output from `analysis.validate_study`,
  - structured diagnostics with stable codes/severities/paths.

## Validation Rules (v1)

Hard-fail conditions:

- import cycle detected,
- import path escapes sandbox root,
- unsupported `schema` or `create_model_intent.profile`,
- missing referenced material/load identifiers,
- invalid region references for assignments/BCs/loads,
- profile-specific solver fields invalid (for example nonlinear-only fields on modal profile),
- non-finite or non-physical numeric values where prohibited.

Current runtime-mapped hard failures that should be preserved by study builtins:

- `ANALYSIS_CREATE_MODEL_INVALID_INTENT`
- `ANALYSIS_CREATE_MODEL_PREP_MISMATCH`
- `ANALYSIS_CREATE_MODEL_PREP_REGION_NOT_FOUND`
- `ANALYSIS_RUN_MODAL_INVALID_MODEL`
- `ANALYSIS_RUN_TRANSIENT_INVALID_MODEL`
- `ANALYSIS_RUN_NONLINEAR_INVALID_MODEL`
- `ANALYSIS_RUN_INVALID_OPTIONS`
- `ANALYSIS_RUN_TRANSIENT_INVALID_OPTIONS`
- `ANALYSIS_RUN_NONLINEAR_INVALID_OPTIONS`

Warning-level conditions:

- optional import missing when `required=false`,
- unknown metadata keys,
- deprecated but still accepted additive fields.

Representative error codes:

- `STUDY_SCHEMA_UNSUPPORTED`
- `STUDY_IMPORT_CYCLE`
- `STUDY_IMPORT_PATH_OUT_OF_ROOT`
- `STUDY_PROFILE_INVALID`
- `STUDY_REFERENCE_NOT_FOUND`
- `STUDY_REGION_REFERENCE_INVALID`
- `STUDY_SOLVER_OPTION_INVALID`
- `STUDY_IMPORT_MERGE_CONFLICT`

## Composability / Includes

v1 include model:

- `imports` is an ordered list of documents to merge into the root study.
- merge semantics are deterministic:
  - scalar/object keys: later imports override earlier imports; root study overrides all imports,
  - keyed list fields merge by stable key,
  - unkeyed list fields append in order.

Keyed list merge keys (v1):

- `model.materials`: `material_id`
- `model.material_assignments`: `region_id`
- `model.boundary_conditions`: `bc_id`
- `model.loads`: `load_id`
- `model.steps`: `step_id`

Merge conflict rules:

- duplicate keyed entry with additive-compatible values: merge,
- duplicate keyed entry with incompatible structural type or `kind`: hard fail (`STUDY_IMPORT_MERGE_CONFLICT`),
- duplicate import path in a single resolution chain: deduplicate by normalized absolute path.

Required guards:

- cycle detection for imports,
- path sandboxing (no escaping study root unless explicitly enabled),
- conflict diagnostics when merge behavior is ambiguous.

This enables shared libraries (materials/policies/load cases) across many studies.

## Canonical Internal Execution Sequence

`analysis.run_study` should resolve to:

1. `geometry.load/v1`
2. `geometry.prep_for_analysis/v1`
3. `analysis.create_model/v1`
4. `analysis.validate/v1`
5. `analysis.run_*/v1` (based on `profile`)
6. `analysis.results/v1`
7. `analysis.trends/v1` (when available)
8. readiness/report generation

This sequence is runtime-owned and stable; study authors do not manually script low-level op choreography.

## Artifact Layout (Normative)

Analysis-level files (stable across runs for the same analysis lineage):

- `.artifacts/analysis/by-id/<analysis_id>/meta.json`
  - analysis metadata (`analysis_id`, `study_id`, `study_fingerprint`, creation/update timestamps).

- `.artifacts/analysis/by-id/<analysis_id>/latest.json`
  - pointer metadata to latest run,
  - recommended fields: `analysis_id`, `latest_run_id`, `updated_at`, `status`.

- `.artifacts/analysis/by-study/<study_fingerprint>/pointer.json`
  - maps study fingerprint to active `analysis_id` for collision-safe discovery.

- `.artifacts/analysis/index/analyses.jsonl`
  - append-only index for quick listing/filtering.

Run-level files (immutable per run):

- `.artifacts/analysis/by-id/<analysis_id>/runs/<run_id>/study_lock.json`
- `.artifacts/analysis/by-id/<analysis_id>/runs/<run_id>/run_manifest.json`
- `.artifacts/analysis/by-id/<analysis_id>/runs/<run_id>/analysis_results.json`
- `.artifacts/analysis/by-id/<analysis_id>/runs/<run_id>/analysis_diagnostics.json`
- `.artifacts/analysis/by-id/<analysis_id>/runs/<run_id>/analysis_trends.json`
- `.artifacts/analysis/by-id/<analysis_id>/runs/<run_id>/readiness_verdict.json`
- `.artifacts/analysis/by-id/<analysis_id>/runs/<run_id>/run.data/`

- `.artifacts/analysis/index/runs.jsonl`
  - append-only run index (`analysis_id`, `run_id`, `status`, timestamps, profile).

ID and collision policy:

- `analysis_id` and `run_id` are generated opaque ids (ULID/UUIDv7 style),
- `study_id` is descriptive metadata only and must not be used as directory key,
- `study_fingerprint` (from resolved `study_lock`) is the stable key for by-study discovery.

## Discovery and Loading

Callers should avoid manual path construction. Preferred discovery uses helper APIs:

- `analysis.resolve_study(study_path)` -> `{ analysis_id, study_fingerprint, root }`
- `analysis.latest_run(analysis_id)` -> `{ run_id, status, updated_at }`
- `analysis.open_run(analysis_id, run_id)` -> run artifact metadata/paths
- `analysis.open_run_data(analysis_id, run_id)` -> `Dataset` handle for `run.data`

These helpers are thin wrappers over on-disk index/pointer files.

Direct-file fallback (no helper builtins):

1. resolve `analysis_id` from `.artifacts/analysis/by-study/<study_fingerprint>/pointer.json`,
2. resolve `run_id` from `.artifacts/analysis/by-id/<analysis_id>/latest.json` (or explicit run selection),
3. load run artifacts from `.artifacts/analysis/by-id/<analysis_id>/runs/<run_id>/...`.

## Study Fingerprint Canonicalization (v1)

`study_fingerprint` is computed from resolved study intent, not volatile run metadata.

- source payload: fully resolved study document after import merge,
- excluded fields: runtime-generated fields (`analysis_id`, `run_id`, timestamps, artifact paths),
- canonicalization: deterministic JSON encoding with sorted keys and UTF-8 bytes,
- hash: `sha256` over canonical bytes,
- storage key format: `sha256_<hex>` used under `by-study/`.

This keeps discovery stable across reruns while changing when reviewed study intent changes.

## `.m` Script Examples

### `scripts/run_simulation.m`

```matlab
function run_simulation(study_path)
  if nargin < 1
    study_path = 'study.yaml';
  end

  validation = analysis.validate_study(study_path);
  if isfield(validation, 'error_code')
    error('Study validation failed: %s (%s)', validation.message, validation.error_code);
  end

  run = analysis.run_study(study_path);
  if isfield(run, 'error_code')
    error('Study run failed: %s (%s)', run.message, run.error_code);
  end

  fprintf('run_id=%s\n', string(run.data.run_id));
  fprintf('status=%s publishable=%d\n', string(run.data.run_status), run.data.publishable);
end
```

### `scripts/inspect_results.m` (helper-based pattern)

```matlab
function inspect_results(study_path, run_id)
  if nargin < 1
    study_path = 'study.yaml';
  end

  resolved = analysis.resolve_study(study_path);
  analysis_id = resolved.data.analysis_id;
  if nargin < 2
    latest = analysis.latest_run(analysis_id);
    run_id = latest.data.run_id;
  end

  run_meta = analysis.open_run(analysis_id, run_id);
  results_path = string(run_meta.data.analysis_results_path);
  dataset_path = string(run_meta.data.run_data_path);

  results = jsondecode(fileread(results_path));
  ds = data.open(dataset_path);

  fprintf('Run status: %s\n', string(results.data.run_status));
  fprintf('Publishable: %d\n', results.data.publishable);

  fprintf('Field count: %d\n', results.data.summary.field_count);
  fprintf('Total elements: %d\n', results.data.summary.total_elements);

  if ds.has_array('displacement')
    u = ds.array('displacement').read();
    fprintf('Max |u| (from run.data): %.6e\n', max(abs(u(:))));
  end
  if ds.has_array('von_mises')
    vm = ds.array('von_mises').read();
    fprintf('Max von Mises (from run.data): %.6e\n', max(vm(:)));
  end

  % Optional desktop field hooks:
  % visualize_field_3d(ds, 'displacement');
  % visualize_field_3d(ds, 'von_mises');
end
```

## Output Evidence Contract (v1)

Generated artifacts include JSON envelopes and a native `.data` dataset under `.artifacts/analysis/by-id/<analysis_id>/runs/<run_id>/`:

- `run.data`
  - primary script/plot/desktop inspection substrate,
  - managed via `data.*`, `Dataset.*`, and `DataArray.*` builtins,
  - required arrays are run-kind-specific (see table below),
  - optional arrays may include mode/snapshot/diagnostic series views.

- `study_lock.json`
  - resolved study after include merge,
  - resolved options/profile,
  - hashes of root study + imported fragments,
  - schema (`runmat-analysis-study-lock/v1`) and builtin version metadata.

- `run_manifest.json`
  - run id, trace id, request id, timings,
  - operation versions executed.

- `analysis_results.json`
  - `analysis.results/v1` envelope payload,
  - governance/CI summary/status/provenance surface,
  - should not be treated as the primary field-array inspection API.

- `analysis_diagnostics.json`
  - structured diagnostic list.

- `analysis_trends.json`
  - trend summary when available.

- `readiness_verdict.json`
  - readiness status + reason codes.

### `run.data` expected shape (v1)

Dataset-level expectations:

- format uses current `DataManifest` shape (`manifest.json` with `arrays` and `attrs`),
- `attrs` should include `run_id`, `study_id`, `profile`, `op_version`, and `created_at`,
- array naming remains stable across runs for script portability.

Required arrays by run kind:

| Run kind | Required | Arrays |
| --- | --- | --- |
| linear static | yes | `displacement`, `von_mises` |
| modal | yes | `modal_eigenvalues_hz`, `modal_residual_norms` |
| transient | yes | `transient_time_points_s`, `transient_residual_norms` |
| nonlinear | yes | `nonlinear_load_factors`, `nonlinear_residual_norms`, `nonlinear_iteration_counts` |

Optional arrays by run kind:

- modal: mode-shape arrays (for example `modal_mode_shape_<index>`),
- transient: displacement snapshot arrays,
- nonlinear: displacement snapshot arrays and nonlinear line-search/tangent-rebuild series.

This keeps field/series inspection on the RunMat data plane while JSON envelopes stay governance-focused.

## Provenance Model

Two complementary layers:

- authoring provenance: revisioned `study.yaml`, includes, and `.m` scripts,
- execution provenance: generated lock + JSON envelopes + `run.data` dataset under immutable run roots.

This aligns with cloud-revisioned `.m` and keeps runtime replay/audit strong.

## Review Checklist (v1)

- study is readable and intent-aligned,
- imports are minimal, deterministic, and conflict-free,
- materials/loads/BCs match engineering expectation,
- solver/profile choices match risk posture,
- no hidden threshold relaxation in imported policy fragments,
- run evidence includes trace/run/version metadata.

## Implementation Notes (`serde` + type registry)

- model study docs as strongly typed `serde` structs/enums,
- support YAML and JSON decode through shared type model,
- expose schema metadata in the runtime type registry for introspection/help,
- make parser diagnostics typed and actionable (`STUDY_IMPORT_CYCLE`, `STUDY_FIELD_INVALID`, etc.),
- keep forward compatibility by tolerating additive fields where safe.

## Immediate Next Step

Implement a minimal `analysis.validate_study` + `analysis.run_study` vertical slice for one linear-static study,
including include-resolution, lock generation, and standard output evidence emission.
