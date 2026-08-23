# Meshing

RunMat turns exact CAD geometry into a deterministic, solver-ready tetrahedral mesh. The result is
not just a list of points and tetrahedra: it is a validated artifact containing boundary topology,
regions, material ownership, interfaces, contacts, field mappings, the fully resolved meshing
request, and a canonical identity. A separate evidence artifact records what every stage checked
and the resources it used.

Use meshing in either of two ways:

| Goal | Entry point |
| --- | --- |
| Generate and inspect a mesh directly | `runmat mesh part.step` |
| Mesh as part of an analysis | Add a `mesh:` block to a `.fea` study and run `runmat run study.fea` |
| Integrate meshing into another host | Use the canonical meshing request, DAG, and execution-host APIs from Rust |

The direct CLI and `.fea` paths import exact STEP, IGES, or Open CASCADE B-rep geometry. They
require a RunMat build with the native Open CASCADE (OCCT) capability. Official native CLI and
Desktop builds include it.

## Quick Start

Given `bracket.step`, generate a linear tetrahedral mesh in metres:

```sh
runmat mesh bracket.step \
  --target-size 0.005 \
  --deviation 0.00005 \
  --material aluminum_6061
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
  --material aluminum_6061 \
  --max-elements 2000000 \
  --seed 42 \
  --json > artifacts/bracket.mesh-summary.json
```

Existing output files are protected. Pass `--force` only when replacing both outputs is intended.
RunMat writes each file atomically, decodes it again, validates it, and verifies that the evidence
and mesh identities agree before reporting success.

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

The deliberately large target size keeps this small fixture quick. Use dimensions appropriate to
your own model in production.

## Geometry Admission

Meshing consumes authoritative geometry, not display triangles. For exact CAD, RunMat imports the
B-rep topology and its curve, surface, pcurve, trim-classifier, and mass-property evaluators. The
geometry revision binds:

- source bytes and format;
- source units and normalization to metres;
- importer and exact-kernel versions;
- model tolerance and bounded healing policy;
- persistent topology mapping; and
- assemblies, bodies, shells, faces, coedges, edges, vertices, interfaces, and contacts.

Display tessellation remains a revision-keyed visualization cache. It is useful for rendering and
selection, but it never becomes the authority for exact curve, surface, or volume meshing.

The `runmat mesh` command interprets source coordinates as metres. A `.fea` study declares source
units explicitly, and RunMat normalizes the imported geometry before deriving its revision:

```yaml
geometry:
  path: ../geometry/bracket.step
  units: millimeter
```

Valid unit names include the values accepted by the geometry document schema, such as `meter` and
`millimeter`. Paths in `.fea` files are resolved relative to the document. If the CAD file is
changed, re-exported, imported with different units, or admitted under a different exact-kernel ABI,
it receives a different geometry revision and cannot accidentally reuse an incompatible mesh.

Geometry admission fails closed when topology is inconsistent, healing would exceed its bound,
exact evaluators are missing, or the worker does not provide the required kernel capability. It
does not replace exact CAD with a bounding box or a display tessellation.

## Meshing Controls

`runmat mesh --help` is the command-line authority. Its principal controls are:

| Option | Meaning | Default |
| --- | --- | --- |
| `--target-size <metres>` | Global target edge length used to construct the metric field | `0.01` |
| `--deviation <metres>` | Maximum curve and surface chordal deviation | `0.0001` |
| `--element-order tet4\|tet10` | Linear or quadratic tetrahedra | `tet4` |
| `--material <id>` | Material assigned to every imported volume region | `material` |
| `--max-elements <count>` | Hard element ceiling | `10000000` |
| `--seed <integer>` | Deterministic tie-breaking seed | `0` |
| `--output <path>` | Solver mesh destination | Source name with `.solver-mesh.cbor` |
| `--evidence <path>` | Evidence destination | Source name with `.meshing-evidence.cbor` |
| `--json` | Print a machine-readable summary | Off |
| `--force` | Atomically replace existing destinations | Off |

Smaller target sizes usually create more elements. Deviation is independent: curved geometry may
need boundary edges much shorter than the global target to stay within the geometric error bound.
Tet10 adds midside nodes, projects boundary midside nodes through exact evaluators, and validates
positive curved Jacobians; it is more expensive than Tet4 but is often preferable for curved
geometry and higher-order solvers.

The seed is part of the logical request. Keep it stable for reproducible artifacts. Changing the
seed is an explicit request change, not a way to bypass a geometry, quality, or budget failure.

`--max-elements` is a hard limit. RunMat does not exceed it and then truncate the mesh. If topology
and quality cannot both be satisfied within the limit, generation fails without publishing a
solver-ready result.

### Meshing in a `.fea` study

The concise study schema exposes the controls most analysis users need:

```yaml
version: 1
kind: study
id: bracket_static

geometry:
  path: ../geometry/bracket.step
  units: millimeter

model:
  profile: linear_static_structural

mesh:
  element_order: tet10
  maximum_elements: 2000000
  target_edge_length_m: 0.005
  maximum_chordal_deviation_m: 0.00005
  maximum_grading_ratio: 1.3
  deterministic_seed: 42

run:
  backend: cpu
```

Check and run it with:

```sh
runmat check studies/bracket_static.fea
runmat run studies/bracket_static.fea
```

The `mesh:` block is resolved into the same canonical request used by direct generation. Algorithm
identities and implementation choices are recorded as evidence; they are not user-selectable
backends. `maximum_grading_ratio` limits how quickly the requested size may change through adjacent
regions. Lower values produce smoother transitions and can require more elements.

For a linear-static profile, omitting `mesh:` uses the defaults shown above. Declare it explicitly
when artifact identity and reviewability matter, especially in CI.

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
  -> refinement and sliver treatment
  -> Tet10 elevation when requested
  -> independent validation
  -> canonical serialization and atomic publication
```

Edge discretizations are shared by every adjacent face, including exact curve parameters and the
separate pcurve images needed at seams. Faces are triangulated in parameter space with constructive
trim recovery and exact/adaptive predicates. The joined surface must be stitched by shared identity,
not coordinate welding, before it can define the protected boundary complex.

Volume generation uses one general constrained-Delaunay tetrahedralizer. It constructively recovers
authored segments and facets, blocks carving at those facets, and assigns retained tetrahedra to
exactly one physical region. A connected constrained volume stays one cohesive work unit; RunMat
does not split it into overlapping submeshes and weld coordinates afterward. Disconnected bodies or
other contractually independent components may run separately and join in canonical identity order.

Every accepted topology mutation preserves recovered constraints, positive orientation, region
ownership, and source provenance. The final validator rechecks the artifact independently of the
generator before publication.

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
and unit. Successful evidence cannot contain a failed invariant. Important quality concepts include:

| Statistic or invariant | Interpretation |
| --- | --- |
| Chordal deviation | Distance between the discrete boundary and its exact curve or surface |
| Normal deviation | Angular difference between discrete and exact surface normals |
| Metric edge length | Edge length measured in the resolved isotropic or anisotropic sizing field |
| Radius-edge ratio | Tetrahedron circumsphere radius relative to its shortest edge; lower is better |
| Scaled Jacobian | Orientation and shape measure normalized to `[-1, 1]`; accepted elements are positive and must meet the request |
| Recovered constraints | Every authored PLC segment and facet is present with orientation and provenance |
| Region classification | Every retained tetrahedron belongs to exactly one material/physical region |

Do not compare only element counts when assessing two meshes. A finer count can still hide poor
elements, boundary error, missing topology, or a changed request. Compare the canonical request,
geometry revision, invariants, achieved distributions, and solver convergence together.

The CBOR evidence file is the durable record. Its identity is bound to the solver mesh, request,
geometry revision, algorithm set, seed, platform capability cohort, and per-stage result identities.
The JSON printed by the CLI is a convenient projection, not a replacement for that artifact.

## Determinism and Caching

Canonical entity ordering, partition identities, exact/adaptive predicates, stable tie-breaking, and
deterministic joins make artifact identity independent of task completion order. The final digest is
computed from canonical content and excludes physical paths, process IDs, worker IDs, timestamps,
and wall-clock measurements.

Stage results are immutable, content-addressed manifests over bounded objects. A cached result is
accepted only when its complete logical identity matches and it independently revalidates. Partial
or invalid output cannot satisfy a dependency or be published as a solver mesh.

These rules have two practical effects:

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

Remote meshing is exposed to native host integrations through the same canonical stage requests and
the general remote pool. A remote worker must advertise the matching meshing algorithm, exact CAD
kernel and ABI, platform, memory/scratch class, and requested element-order capabilities. Geometry,
stage objects, progress, and results remain application-encrypted; the Server sees only coarse run,
allocation, quota, retention, and ciphertext metadata.

Operators prepare capacity with the ordinary RunMat cluster workflow:

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

The standalone `runmat mesh` command currently targets local native workers and does not accept a
`--cluster` option. Product hosts that enable remote meshing supply their existing execution target,
cluster, queue, trust, and worker policy to the meshing DAG executor. Do not use `runmat job submit`
with a CAD pathname as a substitute: that command submits a packaged RunMat program, and a local CAD
path is not implicitly uploaded with it.

This boundary is intentional and useful when integrating RunMat: placement context must remain
outside `MeshingRequest`, so running the same logical request locally or on a compatible remote pool
does not change the canonical mesh identity.

## Solver and Runtime Integration

The published `SolverMeshArtifact` contains everything downstream solvers need:

- stable nodes with coordinates, source provenance, and exact curve/surface parameters;
- Tet4 or Tet10 volume elements and material/region ownership;
- element neighbors;
- Tri3 or Tri6 boundary faces and Line2 or Line3 boundary edges;
- conformal material interfaces and independent nonconformal contact pairs;
- ordered field-topology maps; and
- the geometry revision, resolved request, root stage identity, and canonical digest.

When a `.fea` study does not provide an existing internal mesh artifact, the runtime asks its
installed meshing provider to generate one. It persists the mesh and evidence beneath the configured
`[runtime.fea].artifact_root`, using content-derived names under the meshing store. Before solver
use, the runtime decodes the canonical artifact, validates its complete topology and request, checks
the evidence binding, verifies geometry compatibility, and projects only then into solver options.

Materials and regions cross the boundary by persistent identity. Boundary selectors such as
`face_000001` resolve to the imported exact face identity returned by meshing, so loads and boundary
conditions are not assigned by nearest triangle or centroid.

## Failures, Budgets, and Cancellation

Meshing fails without publishing a solver-ready artifact when it cannot prove the requested result.
Failure categories distinguish invalid geometry, healing limits, unsatisfiable constraints, sizing
conflicts, unreachable quality, node/element/memory/scratch/time/artifact/search/recursion/iteration
budgets, cancellation, numerical failure, and internal invariant violations.

Useful first responses are:

| Failure | What to check |
| --- | --- |
| Invalid geometry | Inspect CAD topology, units, open shells, duplicate entities, and exporter diagnostics |
| Healing limit exceeded | Repair or re-export the CAD model; do not simply inflate a global tolerance |
| Unsatisfiable constraints | Inspect acute features, thin gaps, intersections, and model tolerance |
| Element or memory budget exceeded | Increase the relevant operational budget or make the requested size/deviation less demanding |
| Quality target unreachable | Inspect the reported entities and geometric witnesses; repair geometry or revise an explicit quality target |
| Capability mismatch | Use a native worker with the required OCCT ABI and element-order support |
| Cancelled | Re-run the same request when ready; no partial result is admitted to the production cache |

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
- Keep a hard element ceiling in CI. Treat a budget failure as information instead of accepting a
  silently degraded mesh.
- Save both CBOR artifacts and the JSON summary with analysis results. The mesh alone does not carry
  measured stage evidence.
- Use stable seeds and explicit settings for regression tests. Compare canonical digests only when
  geometry, request, and supported capability cohort are intended to match.
- Validate engineering conclusions with mesh-convergence studies: reduce target size and chordal
  deviation systematically and confirm that quantities of interest converge.

Meshing proves that a discretization satisfies its declared geometric and numerical contracts. It
does not prove that loads, constraints, material data, solver formulation, or a single chosen mesh
are sufficient for the engineering decision. See [Verification & Validation](/docs/fea/validation)
and [Results & Trust](/docs/fea/trust) for the rest of that workflow.
