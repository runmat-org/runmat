# Meshing

RunMat turns exact CAD geometry into a deterministic, solver-ready tetrahedral mesh. The resulting
artifact contains nodes, tetrahedra, boundary topology, persistent regions, interfaces,
contacts, field mappings, the resolved meshing request, and a canonical identity. A companion
evidence artifact records each stage's checks and resource usage.

Meshing has four entry points:

| Goal | Entry point |
| --- | --- |
| Mesh as part of an analysis | Declare `mesh <id> on geometry.<id>` in typed `.fea` source and run `runmat fea run study.fea` |
| Generate canonical artifacts directly from a shell | `runmat mesh part.step` |
| Generate, inspect, and display a mesh in RunMat code | `part = geometry.load(...); solverMesh = meshing.generate(part); mesh(solverMesh)` |
| Integrate meshing into another host | Use the canonical meshing request, DAG, and execution-host APIs from Rust |

The direct CLI and `.fea` paths import exact STEP, IGES, or Open CASCADE B-rep geometry. They
require a RunMat build with the native Open CASCADE (OCCT) capability. Official native CLI and
Desktop builds include it.

## Quick start: mesh a study

Create `bracket.fea` beside a `geometry` directory containing `bracket.step`:

```fea
geometry bracket {
    file = "./geometry/bracket.step"
    units = mm

    region body : solid {
        select id "solid_000001"
        expect exactly_one
    }
}

mesh bracket on geometry.bracket {
    tetrahedra {
        order = quadratic
        target_size = 5 mm
        maximum_deviation = 0.05 mm
        maximum_grading_ratio = 1.3
        deterministic_seed = 42
    }
}

material aluminum {
    mass_density = 2700 kg/m^3
    elastic { youngs_modulus = 68.9 GPa poisson_ratio = 0.33 }
}

model bracket_model {
    assign material.aluminum to geometry.bracket.region.body
}

study bracket_static {
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.bracket_model
    formulation = experimental
    analysis static { structural static {} }
}
```

The geometry declaration admits exact CAD and gives its coordinates physical units. The mesh
declaration says how to discretize that geometry. The study selects the mesh; material ownership is
declared separately on the solid region. Check the complete source before running it:

```sh
runmat fea check bracket.fea
runmat fea plan bracket.fea
runmat fea run bracket.fea
```

`check` resolves modules, types, units, regions, and declaration contracts. `plan` reports the
selected geometry, mesh, model, execution policy, and logical work. `run` materializes the exact
geometry, generates or resolves the canonical solver mesh, compiles the model, and executes the
admitted analysis graph.

For a linear mesh, change only `order = linear`. Omit `maximum_grading_ratio` or
`deterministic_seed` when their defaults are acceptable; the geometry, mesh, and model bindings
remain required. The direct CLI and programmatic API expose a maximum-element admission cap; the
current `.fea` mesh declaration does not.

## Programmatic and advanced mesh inspection

RunMat code is useful when you need to inspect topology and quality, display several region or
section views, or process mesh arrays before constructing an analysis object:

```matlab
part = geometry.load("bracket.step", "Units", "millimeter");
mesh(part);

solverMesh = meshing.generate(part, ...
    "TargetSize", 0.005, ...
    "MaximumDeviation", 0.00005, ...
    "MaximumGradingRatio", 1.3, ...
    "ElementOrder", "tet10", ...
    "MaximumElements", uint64(2000000), ...
    "Seed", uint64(42));

mesh(solverMesh);
```

`geometry.load` returns a `geometry.Asset`: admitted geometry in metres, exact CAD authority when
the source is STEP, IGES, or BREP, persistent topology, and render tessellation. `mesh(part)` is a
view of that geometry; it does not generate solver elements. `meshing.generate` returns an immutable
`meshing.Mesh` whose canonical solver artifact and matching evidence have already passed joint
validation. `mesh(solverMesh)` displays its solver boundary. These overloads use RunMat's object
dispatch; ordinary numeric `mesh(X, Y, Z)` behavior is unchanged.

Geometry display options are visual only:

```matlab
mesh(part, "Edges", true, "XRay", false);
```

Solver-mesh display can select physical volume regions or cut a section. Region identifiers come
from `meshing.regions`, not material names:

```matlab
regions = meshing.regions(solverMesh);
firstRegionInfo = regions{1};
firstRegion = firstRegionInfo.id;
mesh(solverMesh, "Regions", firstRegion);

nodeData = meshing.nodes(solverMesh);
coordinates = nodeData.coordinates;
xMid = (min(coordinates(:, 1)) + max(coordinates(:, 1))) / 2;
mesh(solverMesh, "Section", [1 0 0 -xMid], "Edges", true);
```

The section vector is the plane `[a b c d]` in metres, with `a*x + b*y + c*z + d = 0`.

### Inspect topology, quality, and evidence from RunMat code

The inspection functions return script-friendly data while preserving topology identifiers as
typed `uint64` values:

```matlab
nodes = meshing.nodes(solverMesh);
elements = meshing.elements(solverMesh);
faces = meshing.boundaryFaces(solverMesh);
regions = meshing.regions(solverMesh);

quality = meshing.quality(solverMesh);
disp(quality.minimum_corner_scaled_jacobian);
disp(quality.p05_corner_scaled_jacobian);
disp(quality.maximum_corner_aspect_ratio);
disp(quality.p95_corner_aspect_ratio);
disp(quality.total_corner_volume_m3);

evidence = meshing.evidence(solverMesh);
disp(evidence.artifact_digest);
disp(evidence.generated_elements);
disp(evidence.wall_time_ms);
disp(evidence.stages);
```

`quality.corner_scaled_jacobian`, `quality.corner_aspect_ratio`, and
`quality.corner_volume_m3` are per-element column vectors aligned with `quality.element_ids`. They
describe each tetrahedron's four corner nodes. For Tet4 they describe the element itself; for Tet10,
the bound evidence is the authority for certified curved isoparametric Jacobians and achieved
high-order geometry checks. The summary fields make quick corner-quality checks convenient. The
evidence also remains authoritative for admission-stage invariants, achieved distributions, and
resource measurements. `evidence.resources` and each record in `evidence.stages` retain counters as
typed `uint64` values; stage records include progress, memory/scratch peaks, search and iteration
work, elapsed time, cancellation checkpoints, and invariant results.

Save and reload the jointly validated pair when a script needs explicit artifact paths:

```matlab
meshing.save(solverMesh, ...
    "artifacts/bracket.solver-mesh.cbor", ...
    "artifacts/bracket.meshing-evidence.cbor");

loadedMesh = meshing.load( ...
    "artifacts/bracket.solver-mesh.cbor", ...
    "artifacts/bracket.meshing-evidence.cbor");
```

Existing destinations are protected. Add `"Force", true` only for intentional replacement. Files
are staged before either destination is replaced, and a publication error rolls the pair back;
`meshing.load` decodes both and rejects a mismatched or invalid pair.

### Keep meshing and physical assignments separate

Meshing owns geometry discretization and persistent regions. FEA owns materials, assignments,
excitations, constraints, natural boundaries, solver configuration, fields, and result visualization:

```fea
model bracket_model {
    assign material.steel to geometry.bracket.region.body
}
```

The study compiles model assignments only after exact meshing has projected persistent CAD regions
onto canonical elements. Material assignment is therefore by typed geometry region and never an
option to `meshing.generate` or the `mesh` declaration.

RunMat code does not choose local versus remote placement in the meshing request. The installed
native host supplies its existing execution provider, pool, queue, trust, retry, cancellation, and
artifact-transfer policy. The same code therefore uses local parallel workers on a local host and
compatible distributed capacity when the product host is configured for it.

### Command-line generation

Given `bracket.step`, generate a linear tetrahedral mesh in metres:

```sh
runmat mesh bracket.step \
  --target-size 0.005 \
  --deviation 0.00005
```

This writes two files beside the source:

- `bracket.solver-mesh.cbor` — the solver-ready mesh artifact
- `bracket.meshing-evidence.cbor` — independently bound stage and quality evidence

Use explicit destinations in scripts and CI:

```sh
mkdir -p artifacts
runmat mesh geometry/bracket.step \
  --output artifacts/bracket.mesh.cbor \
  --evidence artifacts/bracket.mesh-evidence.cbor \
  --target-size 0.005 \
  --deviation 0.00005 \
  --element-order tet10 \
  --max-elements 2000000 \
  --seed 42 \
  --json > artifacts/bracket.mesh-summary.json
```

Existing output files are protected. Pass `--force` only when replacing both outputs is intended.
RunMat stages the pair before publication, rolls back publication errors, decodes both outputs,
validates them, and verifies that the evidence and mesh identities agree before reporting success.

### Run the repository fixture

When building RunMat from this repository, the checked-in B-rep box is a self-contained smoke test:

```sh
cargo run -p runmat --features occt-native -- mesh \
  crates/runmat-geometry/io/tests/fixtures/box.brep \
  --target-size 10 \
  --deviation 0.1 \
  --max-elements 10000 \
  --output /tmp/runmat-box.mesh.cbor \
  --evidence /tmp/runmat-box.evidence.cbor \
  --json
```

The large target size keeps this small fixture quick. Use dimensions appropriate to your own model
in production.

## Geometry Admission

The exact CAD representation is the authority for meshing. RunMat imports the B-rep topology and
its curve, surface, pcurve, trim-classifier, and mass-property evaluators. The geometry revision
binds:

- source bytes and format;
- source units and normalization to metres;
- importer and exact-kernel versions;
- model tolerance and bounded healing policy;
- persistent topology mapping; and
- assemblies, bodies, shells, faces, coedges, edges, vertices, interfaces, and contacts.

Display tessellation remains a revision-keyed cache for rendering and selection. Exact curve,
surface, and volume meshing operate on the B-rep and its evaluators.

`runmat mesh` accepts `--source-units meter|millimeter|inch` when source geometry does not declare
its units. A `.fea` geometry declaration always states imported coordinate units explicitly:

```fea
geometry bracket {
    file = "../geometry/bracket.step"
    units = mm
}
```

Paths in `.fea` files are resolved relative to the importing module and confined to the frozen
project. If the CAD file is
changed, re-exported, imported with different units, or admitted under a different exact-kernel ABI,
the geometry revision changes, preventing reuse of an incompatible mesh.

Geometry admission fails closed when topology is inconsistent, healing would exceed its bound,
exact evaluators are missing, or the worker does not provide the required kernel capability. RunMat
does not substitute a bounding box or display tessellation for the exact CAD model.

## Meshing Controls

`runmat mesh --help` is the command-line authority. Its principal controls are:

| Option | Meaning | Default |
| --- | --- | --- |
| `--source-units meter\|millimeter\|inch` | Coordinate units for sources that do not declare them | `meter` |
| `--target-size <metres>` | Global target edge length used to construct the metric field | `0.01` |
| `--deviation <metres>` | Maximum curve and surface chordal deviation | `0.0001` |
| `--element-order tet4\|tet10` | Linear or quadratic tetrahedra | `tet4` |
| `--max-elements <count>` | Hard element ceiling | `10000000` |
| `--seed <integer>` | Deterministic tie-breaking seed | `0` |
| `--output <path>` | Solver mesh destination | Source name with `.solver-mesh.cbor` |
| `--evidence <path>` | Evidence destination | Source name with `.meshing-evidence.cbor` |
| `--json` | Print a machine-readable summary | Off |
| `--force` | Atomically replace existing destinations | Off |
| `--cluster <id>` | Execute through the named remote cluster | Local execution |
| `--project <id>` | Remote project; otherwise use the active project | Active project |
| `--queue <name>` | Remote scheduler queue | `default` |
| `--workers <count>` | Remote worker capacity; zero selects the meshing minimum | `0` |
| `--trust-identity <fingerprint>` | Pinned remote endpoint identity; required with explicit `--cluster` | None for local execution |
| `--idempotency-key <key>` | Stable identity for safe submission retries | Generated |
| `--detach` | Return after remote submission; materialize verified outputs on `job attach` | Off |

Smaller target sizes usually create more elements. Deviation is independent: curved geometry may
need boundary edges much shorter than the global target to stay within the geometric error bound.
Tet10 adds midside nodes, projects boundary midside nodes through exact evaluators, and validates
positive curved Jacobians. It has higher node and validation costs than Tet4 and is often preferable
for curved geometry and higher-order solvers.

The seed is part of the logical request. Keep it stable for reproducible artifacts. Changing the
seed is an explicit request change, not a way to bypass a geometry, quality, or budget failure.

`--max-elements` is a hard limit. RunMat does not exceed it and then truncate the mesh. If topology
and quality cannot both be satisfied within the limit, generation fails without publishing a
solver-ready result.

### Meshing in a `.fea` study

The typed `.fea` mesh declaration exposes the controls that are part of a study definition:

```fea
mesh bracket on geometry.bracket {
    tetrahedra {
        order = quadratic
        target_size = 5 mm
        maximum_deviation = 0.05 mm
        maximum_grading_ratio = 1.3
        deterministic_seed = 42
    }
}
```

Check and run it with:

```sh
runmat check studies/bracket_static.fea
runmat run studies/bracket_static.fea
```

The complete procedure, constraints, loads, and outputs are in
`docs/fea/examples/structural-static-experimental.fea`. The declaration is resolved into the same canonical request used by direct generation. Algorithm
identities and implementation choices are recorded as evidence; they are not user-selectable
backends. `maximum_grading_ratio` limits how quickly the requested size may change through adjacent
regions. Lower values produce smoother transitions and can require more elements.

`order`, `target_size`, and `maximum_deviation` are required. `maximum_grading_ratio` and
`deterministic_seed` have canonical defaults. The language does not currently expose region-local
refinement, boundary layers, hybrid element families, or user-selectable meshing algorithms.

## What the Mesher Does

The geometric work is an explicit deterministic DAG:

```text
exact geometry admission and bounded healing
  -> sizing-field resolution
  -> authoritative edge batches
  -> global curve join and validation
  -> trimmed-face batches
  -> surface/shell join and protected boundary complex
  -> constrained Delaunay tetrahedralization
  -> segment and facet recovery
  -> region/void carving
  -> solver-coordinate materialization and protected legalization
  -> size and radius-edge refinement
  -> scaled-Jacobian sliver optimization
  -> Tet10 elevation when requested
  -> independent validation
  -> canonical serialization and atomic publication
```

Adjacent faces share edge discretizations, including exact curve parameters and the separate pcurve
images needed at seams. Faces are triangulated in parameter space with constructive trim recovery
and exact/adaptive predicates. The surface join uses shared topology identities instead of
coordinate welding before constructing the protected boundary complex.

Volume generation uses one general constrained-Delaunay tetrahedralizer. It constructively recovers
authored segments and facets, blocks carving at those facets, and assigns retained tetrahedra to
exactly one physical region. A connected constrained volume stays one cohesive work unit; RunMat
does not split it into overlapping submeshes and weld coordinates afterward. Disconnected bodies or
other contractually independent components may run separately and join in canonical identity order.

Every accepted topology mutation preserves recovered constraints, positive orientation, region
ownership, and source provenance. The final validator rechecks the artifact independently of the
generator before publication.

Constraint recovery retains exact constructed-point representations while it proves authored edge
and facet incidence. Once those facets are protected, RunMat materializes the coordinates stored in
the solver artifact and re-legalizes only interior topology; protected boundary facets and region
interfaces cannot be crossed. Refinement then inserts points for metric edge-length and radius-edge
violations. A separate optimization pass handles scaled-Jacobian slivers through bounded local
relocation or deletion. Keeping those stages separate prevents a shape threshold from driving an
unbounded insertion loop, while the final quality gate still requires every configured bound.

## Quality and Statistics

A successful mesh has passed topology, geometry, sizing, quality, provenance, resource, and
serialization checks. The short terminal summary reports node, tetrahedron, and boundary-face
counts. For automation, capture JSON:

```sh
runmat mesh bracket.step --target-size 0.005 --json > mesh-summary.json
jq '{canonical_digest, node_count, element_count, boundary_face_count}' mesh-summary.json
jq '.resource_usage' mesh-summary.json
```

Inspect per-stage counters, invariant checks, and achieved error distributions:

```sh
jq '.stages[] | {
  stage,
  partition,
  entity_counts,
  invariants,
  achieved_error_distributions,
  completed_work,
  estimated_work,
  peak_memory_bytes,
  peak_scratch_bytes,
  search_work,
  iterations,
  cancellation_checkpoints
}' mesh-summary.json
```

The exact keys in `entity_counts` and `achieved_error_distributions` depend on the stage. Each error
distribution includes a sample count, minimum, mean, 95th percentile, 99th percentile, maximum,
and unit. Successful evidence cannot contain a failed invariant. The evidence reports these quality
measures and invariants:

| Statistic or invariant | Interpretation |
| --- | --- |
| Chordal deviation | Distance between the discrete boundary and its exact curve or surface |
| Normal deviation | Angular difference between discrete and exact surface normals |
| Metric edge length | Edge length measured in the resolved isotropic or anisotropic sizing field |
| Radius-edge ratio | Tetrahedron circumsphere radius relative to its shortest edge; lower is better |
| Scaled Jacobian | Orientation and shape measure normalized to `[-1, 1]`; accepted elements are positive and must meet the request |
| Recovered constraints | Every authored PLC segment and facet is present with orientation and provenance |
| Region classification | Every retained tetrahedron belongs to exactly one persistent geometric region |

Do not compare only element counts when assessing two meshes. A finer count can still hide poor
elements, boundary error, missing topology, or a changed request. Compare the canonical request,
geometry revision, invariants, achieved distributions, and solver convergence together.

The CBOR evidence file is the durable record. Its identity is bound to the solver mesh, request,
geometry revision, algorithm set, seed, platform capability cohort, and per-stage result identities.
The JSON printed by the CLI contains selected evidence fields; retain the CBOR artifact for the
complete record.

## Determinism and Caching

Canonical entity ordering, partition identities, exact/adaptive predicates, stable tie-breaking, and
deterministic joins make artifact identity independent of task completion order. The final digest is
computed from canonical content and excludes physical paths, process IDs, worker IDs, timestamps,
and wall-clock measurements.

Stage results are immutable, content-addressed manifests over bounded objects. A cached result is
accepted only when its complete logical identity matches and it independently revalidates. Partial
or invalid output cannot satisfy a dependency or be published as a solver mesh.

Caching follows the canonical identity:

- the same supported request and geometry can reuse verified work; and
- changing units, tolerances, sizing, seed, element order, source bytes, or relevant build
  capability produces a distinct identity.

Elapsed time and resource measurements live in evidence and do not affect topology identity.

## Parallel and Distributed Meshing

Meshing uses RunMat's general execution system. The mesher owns the geometric DAG, valid partition
boundaries, and deterministic joins. The execution system owns pools, placement, retries,
cancellation, fencing, worker lifecycle, encrypted transport, and artifact transfer. There is no
separate meshing scheduler or meshing-specific cluster configuration.

Independent curve batches run after geometry admission. Independent face batches run after the
global curve join. Disconnected volume components, validation gates, and safe order-elevation work
can also run concurrently. Stage barriers prevent a downstream task from observing a partial join.
Large inputs and results move as verified, manifest-rooted objects rather than being placed in task
messages.

### Local parallel execution

`runmat mesh` automatically executes the DAG through native worker processes and requires no Server:

```sh
runmat mesh bracket.step --target-size 0.005 --json
```

Partitioning is automatic. Do not split CAD faces or bodies by hand; manual sharding can destroy
shared-edge identity and constrained-volume correctness.

### Remote execution

Pass `--cluster` to send the same meshing request through the general remote pool. The source CAD
must be inside a RunMat project so the submitted run can be bound to an exact project revision:

```sh
runmat mesh geometry/bracket.step \
  --target-size 0.005 \
  --cluster engineering \
  --queue cpu \
  --workers 16 \
  --trust-identity sha256:<endpoint-fingerprint>
```

The CLI waits for the Server-assigned run identity before authorizing and encrypting the chunked CAD
closure. The Server never receives the CAD, mesh, evidence, or decryption key as plaintext. Workers
admit the CAD, derive their own exact-kernel and platform identity, and execute the same canonical
stage DAG used locally. A worker must advertise the matching meshing algorithm, exact CAD kernel and
ABI, platform, memory/scratch class, and requested element-order capabilities.

Without `--detach`, the CLI downloads every terminal result object, checks its digest and run
authority, reconstructs and jointly validates the solver mesh and evidence, and only then writes the
two requested local files. `--detach` returns the durable run ID before creating local files. The
CLI records the absolute `--output` and `--evidence` destinations, together with the explicit
`--force` decision, in the run's permission-restricted private local attach state. A later
`runmat job attach RUN_ID` downloads and verifies the same terminal closure before atomically
materializing that pair. Retrying attach is idempotent when those files already contain the exact
verified result; a different or incomplete existing pair is preserved unless the original
submission explicitly authorized replacement with `--force`.

Use `runmat job show`, `runmat job attach`, or `runmat job cancel` with the run ID for lifecycle
control. Because deferred destinations are client-local intent rather than execution identity, the
Server and workers never receive filesystem paths and the logical mesh identity is unchanged.

Operators prepare capacity with the `runmat cluster` workflow:

```sh
runmat login
runmat cluster create --name analysis-workers --queue default --json
runmat cluster enroll <cluster-id> --json
runmat cluster join enroll --token <single-use-token>
runmat cluster join run
```

The last two commands normally run on an execution node. Production deployments usually install the
node as an operating-system service; see [Remote Execution](/docs/execution/remote) for trust,
enrollment, service, queue, and recovery setup.

Placement context remains outside `MeshingRequest`. Running the same logical request locally or on
a compatible remote pool therefore preserves the canonical mesh identity.

## Solver and Runtime Integration

The published `SolverMeshArtifact` provides solver-facing topology and metadata:

- stable nodes with coordinates, source provenance, and exact curve/surface parameters;
- Tet4 or Tet10 volume elements and persistent region ownership;
- element neighbors;
- Tri3 or Tri6 boundary faces and Line2 or Line3 boundary edges;
- conformal region interfaces and independent nonconformal contact pairs;
- ordered field-topology maps; and
- the geometry revision, resolved request, physical validation-manifest reference, and logical
  canonical digest. Legal partition and chunk layouts can have different manifest references while
  retaining the same mesh digest.

When a `.fea` study runs, the native executor asks its installed meshing provider to realize the
declared mesh. It persists the mesh and evidence beneath the configured
`[runtime.fea].artifact_root`, using content-derived names under the meshing store. Before solver
use, the runtime decodes the canonical artifact, validates its complete topology and request, checks
the evidence binding, verifies geometry compatibility, and then projects it into solver options.

Meshing owns region identity, not material assignment. A canonical mesh can therefore be reused by
multiple studies that assign different materials to the same regions without changing mesh
identity. The compiled study resolves typed assignments against each mesh region's persistent
source identity when it constructs solver coefficients. Material declarations never
imply coverage: every active material-bearing element must be covered by exactly one compatible
typed assignment, even when the source declares only one material. Use explicit solid, shell,
beam, and cell region targets for heterogeneous and mixed-dimensional models. Gaps, overlaps, stale
selectors, and formulation mismatches fail before operator assembly. The meshing result carries one
typed topology-alias inventory: `edge_######`, `face_######`, and `solid_######` retain the
importer's source order and persistent occurrence-qualified identity on local and remote paths.
Thus `solid_000002` can select the second imported solid without depending on final element order,
while `face_000001` resolves the exact face used by a load or boundary condition rather than a
nearest triangle or centroid. A face becomes a material-bearing target only when it is deliberately
used as a shell domain.

## Failures, Budgets, and Cancellation

Meshing fails without publishing a solver-ready artifact when it cannot prove the requested result.
Failure categories distinguish invalid geometry, healing limits, unsatisfiable constraints, sizing
conflicts, unreachable quality, node/element/memory/scratch/time/artifact/search/recursion/iteration
budgets, cancellation, numerical failure, and internal invariant violations.

Investigate the reported failure category as follows:

| Failure | What to check |
| --- | --- |
| Invalid geometry | Inspect CAD topology, units, open shells, duplicate entities, and exporter diagnostics |
| Healing limit exceeded | Repair or re-export the CAD model; increasing a global tolerance is not a substitute for repair |
| Unsatisfiable constraints | Inspect acute features, thin gaps, intersections, and model tolerance |
| Element or memory budget exceeded | Increase the relevant operational budget or make the requested size/deviation less demanding |
| Quality target unreachable | Inspect the reported entities and geometric witnesses; repair geometry or revise an explicit quality target |
| Capability mismatch | Use a native worker with the required OCCT ABI and element-order support |
| Cancelled | Re-run the same request; no partial result is admitted to the production cache |

Cancellation is cooperative and checked throughout expensive geometry and meshing work. Local and
remote hosts propagate the existing execution cancellation authority; bounded checkpoints let a
stage clean up without allowing stale or partial publication. Pure content-addressed stages may be
retried after infrastructure loss. Final publication is fenced and is never treated as freely
replayable after an ambiguous effect.

## Practical Guidance

- Start with an explicit geometry unit and a target size based on the smallest feature that matters
  to the physics, not the model's overall bounding box.
- Set chordal deviation from acceptable boundary error. Curved surfaces often control mesh density
  even when the global target is larger.
- Use Tet4 for quick linear studies and Tet10 when curved-boundary accuracy or a quadratic solver
  formulation justifies the additional nodes and validation cost.
- Keep a hard element ceiling in CI. Investigate a budget failure rather than accepting a mesh with
  reduced sizing or quality requirements.
- Save both CBOR artifacts and the JSON summary with analysis results. The mesh alone does not carry
  measured stage evidence.
- Use stable seeds and explicit settings for regression tests. Compare canonical digests only when
  geometry, request, and supported capability cohort are intended to match.
- Validate engineering conclusions with mesh-convergence studies: reduce target size and chordal
  deviation systematically and confirm that quantities of interest converge.

A validated mesh proves that the discretization satisfies its declared geometric and numerical
contracts. Engineering validation must also cover loads, constraints, material data, solver
formulation, and mesh convergence. See [Verification & Validation](/docs/fea/validation) and
[Results & Trust](/docs/fea/trust) for that workflow.
