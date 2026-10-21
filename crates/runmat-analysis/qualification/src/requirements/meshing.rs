//! Meshing requirements: exact CAD import, tetrahedral discretization, real-part success and distributed meshing.
use crate::requirement::*;
const IMPORT: &[Envelope] = cells!(
    [Formulation::SharedInfrastructure],
    [MeshOrder::MeshIndependent],
    [
        ExecutionEnvelope::SourceCheck,
        ExecutionEnvelope::NativeProcess,
        ExecutionEnvelope::PublicStudy
    ]
);
const MESH: &[Envelope] = cells!(
    [Formulation::SharedInfrastructure],
    [MeshOrder::CanonicalMixedTopology],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::NativeProcess,
        ExecutionEnvelope::PublicStudy
    ]
);
const PUBLIC: &[Envelope] = cells!(
    [Formulation::SharedInfrastructure],
    [MeshOrder::CanonicalMixedTopology],
    [ExecutionEnvelope::PublicStudy]
);
const WORKERS: &[Envelope] = cells!(
    [Formulation::SharedInfrastructure],
    [MeshOrder::CanonicalMixedTopology],
    [
        ExecutionEnvelope::NativeProcess,
        ExecutionEnvelope::Durable,
        ExecutionEnvelope::RemoteExecution
    ]
);
const DECOMP: &[Envelope] = cells!(
    [Formulation::SharedInfrastructure],
    [MeshOrder::CanonicalMixedTopology],
    [ExecutionEnvelope::DomainDecomposed]
);
const PARALLEL: &[Envelope] = cells!(
    [Formulation::SharedInfrastructure],
    [MeshOrder::CanonicalMixedTopology],
    [ExecutionEnvelope::ParallelCpu]
);
const FEA_MESH_GEOMETRY_001: Requirement = Requirement {
    id: "FEA-MESH-GEOMETRY-001",
    title: "Exact CAD import of real parts and assemblies",
    stages: &[Stage::Meshing, Stage::Authoring],
    acceptance: "STEP (including AP242 product, assembly, parameter and material data), IGES and BREP import preserve the authoritative B-rep topology, mass properties, units, part structure and authored names, or fail with a typed rejection. Imported geometry that the public examples and website studies use resolves by authored name at check, plan and run alike.",
    owners: &["crates/runmat-geometry/io/src/occt/evaluator.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "exact_formats",
            "Exact STEP, IGES and BREP import keeps authoritative B-rep topology and mass properties, or fails typed; tessellated formats (STL, OBJ, PLY, glTF) are rejected for solver meshing with a typed diagnostic. A governed corpus of at least the verification/meshing/corpus.json fixtures passes with recorded hashes, topology counts and mass properties.",
            EvidenceKind::IndependentReference,
            IMPORT,
        ),
        obligation(
            "multibody_assembly_import",
            "Assemblies with touching, imprinted or nested bodies import as named parts with explicit interfaces and without failing on shared faces: a gear, an inductor and the surrounding air; a core, primary and air; a plate and its coolant volume. Each part keeps its own solid, shell and face identities and the assembly keeps the interface lineage the study selects against.",
            EvidenceKind::SemanticRealization,
            IMPORT,
        ),
        obligation(
            "authored_names_survive_to_run",
            "Face, edge, solid and assembly names authored in the STEP file (for example Beam, Fixed end, Loaded end, Heat sink, Mounting face, Coil terminal in, Coil terminal out, Outer boundary, Inlet, Outlet, Walls) are read from the file and resolve with `select name` at check and at run to the same entities. Unnamed or ambiguous selections fail with a located diagnostic; no run silently selects zero or the wrong entities.",
            EvidenceKind::SemanticRealization,
            IMPORT,
        ),
        obligation(
            "selectors_on_constructed_and_cell_regions",
            "`select id` on constructed geometry, `select result` and `select name` for fluid `cell` regions on CAD, and every other selector form the language admits resolve identically at check and at run; a selector that check accepts cannot fail at run for lack of an entity.",
            EvidenceKind::SemanticRealization,
            IMPORT,
        ),
        obligation(
            "geometry_load_units_option",
            "The `geometry.load` units option and the `units` declaration of a geometry block scale imported coordinates consistently and reject mixed or missing units with a typed diagnostic before meshing.",
            EvidenceKind::RejectionInvariant,
            IMPORT,
        ),
    ],
};
const FEA_MESH_DISCRETIZATION_001: Requirement = Requirement {
    id: "FEA-MESH-DISCRETIZATION-001",
    title: "Valid tetrahedral meshes on exact geometry",
    stages: &[Stage::Meshing],
    acceptance: "One metric size field drives curve, surface and volume discretization on exact geometry, producing conforming Tet4 and curved Tet10 meshes with valid elements, exact boundary and interface faces and recorded quality evidence, inside the authored work, memory and time budgets. Mesh acceptance includes solver relevance, not only element quality.",
    owners: &["crates/runmat-meshing/execution/src/dispatcher.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "curve_discretization",
            "Curve discretization respects exact edge and pcurve domains, tiny edges, periodic curves and seams on real parts, within the declared chordal deviation.",
            EvidenceKind::IndependentReference,
            MESH,
        ),
        obligation(
            "surface_discretization",
            "Constructive trimmed-surface triangulation handles seams, poles, many trims and dense faces within the memory budget, and a dense surface of at least 90,000 nodes and 180,000 triangles is retained and published within cache-entry and codec limits.",
            EvidenceKind::ResourceAdmission,
            MESH,
        ),
        obligation(
            "volume_conformity",
            "Three-dimensional constrained Delaunay recovers segments and facets, carves exterior, nested voids and regions, and preserves conformal interfaces and contact pairs; the exact boundary is matched to the declared tolerance, with a real-part corpus beside the fixtures.",
            EvidenceKind::IndependentReference,
            MESH,
        ),
        obligation(
            "volume_quality_convergence",
            "Volume quality refinement terminates within the authored work and time budget on the bracket, heat sink and tuning fork, ending with every element at or above the declared quality floor, and refinement yields to coupled repair fairly with stale rejections invalidated and true global exhaustion detected.",
            EvidenceKind::ResourceAdmission,
            MESH,
        ),
        obligation(
            "tet10_elevation",
            "Tet10 elevation yields valid curved elements without near-slivers: positive curved Jacobian everywhere and a scaled-Jacobian floor stated per part, including thin airfoil sections and the tuning fork.",
            EvidenceKind::IndependentReference,
            MESH,
        ),
        obligation(
            "sizing_controls",
            "Global, regional, face, edge and feature-based size controls and graded sizing are exposed in the `.fea` mesh block and drive the one metric field; a graded heat-sink surface and a channel-resolving cold-plate mesh are produced from authored controls without manual partitioning.",
            EvidenceKind::SemanticRealization,
            MESH,
        ),
        obligation(
            "solver_relevant_acceptance",
            "Mesh acceptance includes solver patch and convergence relevance on real parts: a recorded refinement series shows the target quantity converging, so coarse Tet4 meshes are flagged where stiffness is not credible.",
            EvidenceKind::SpatialConvergence,
            MESH,
        ),
        obligation(
            "mesh_evidence_artifact",
            "Every mesh ships an evidence artifact with per-stage counters, invariants, quality distributions (minimum, mean, p95, p99, maximum of chordal deviation, normal deviation, metric edge length, radius-edge ratio and scaled Jacobian), peak memory and the seed and digest.",
            EvidenceKind::PublicationIntegrity,
            MESH,
        ),
        obligation(
            "independent_mesher_differential",
            "A differential comparison against an independent mesher (OCCT incremental mesh) on a box, a curved part and a real part records volume, gradient energy and mismatches with their dispositions.",
            EvidenceKind::IndependentReference,
            MESH,
        ),
    ],
};
const FEA_MESH_REALPARTS_001: Requirement = Requirement {
    id: "FEA-MESH-REALPARTS-001",
    title: "Real parts mesh with a published success rate",
    stages: &[Stage::Meshing, Stage::PublicSurface],
    acceptance: "A stated corpus of real parts and every website hero part meshes through the public command on the release build with defaults, with a published success rate and time distribution on stated hardware. Misses are published, not hidden.",
    owners: &["crates/runmat-meshing/execution/src/dispatcher.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "corpus_success_rate",
            "A named corpus (bracket, heat sink, tuning fork, cantilever, blade, lugs, tower, and the website fixtures) meshes on the release build; the success rate and the wall-time and memory distributions are generated from the run and published with hardware, including failures.",
            EvidenceKind::PerformanceMeasurement,
            PUBLIC,
        ),
        obligation(
            "original_study_meshes",
            "`runmat mesh bracket.stp` with bare defaults is accepted; the KiCad heat sink meshes and its steady study closes the ~10 W balance; the tuning fork meshes with Tet10 and returns two modes with eigen-residuals; each stays inside the authored 512 MiB and one hour envelope with whole-task resources measured.",
            EvidenceKind::PublicRun,
            PUBLIC,
        ),
        obligation(
            "website_fixture_parts_mesh",
            "Every STEP fixture the website studies use meshes and passes quality acceptance at the authored size: cold plate (6 x 3 mm channels), enclosure, carotid bifurcation, static mixer, journal bearing, filter, muffler, cabin, speaker hemisphere with diaphragm, room, induction gear with inductor and air, busbar, current sensor, electrode.",
            EvidenceKind::PublicRun,
            PUBLIC,
        ),
        obligation(
            "twisted_and_curved_hero_parts",
            "Twisted or extruded blades, lugs and towers, fused pcurve tolerances, planar curve ranges and booleans on curved STEP mesh without hangs or invalid curved elements.",
            EvidenceKind::PublicRun,
            PUBLIC,
        ),
        obligation(
            "mesh_command_defaults",
            "`runmat mesh` accepts defaults and writes the mesh and its evidence artifact; failures are typed and name the stage.",
            EvidenceKind::PublicRun,
            PUBLIC,
        ),
        obligation(
            "wavelength_and_feature_size_checks",
            "Acoustic and electromagnetic meshes are checked against wavelength or skin-depth targets at check or plan time, and a mesh coarser than the declared elements-per-wavelength rule produces a located diagnostic rather than a silent accuracy loss.",
            EvidenceKind::RejectionInvariant,
            PUBLIC,
        ),
    ],
};
const FEA_MESH_EXECUTION_001: Requirement = Requirement {
    id: "FEA-MESH-EXECUTION-001",
    title: "Deterministic, distributed and bounded meshing",
    stages: &[Stage::Meshing, Stage::Scale],
    acceptance: "Meshing runs on the same execution layer as studies: independent curve, face and volume batches fan out over worker processes, a single connected volume partitions across workers, results are identical across backends, worker counts and platforms, and every stage enforces its budgets and cancellation. The browser never meshes silently.",
    owners: &["crates/runmat-execution-runner-native/tests/meshing_process_conformance.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "worker_batches",
            "Deterministic edge and face batches execute on worker processes without manual partitioning, and the same artifact results from one or many workers.",
            EvidenceKind::ExecutionEquivalence,
            WORKERS,
        ),
        obligation(
            "backend_equivalence",
            "Serial, local-process, durable-local and remote pool runs return the same canonical topology and digest.",
            EvidenceKind::ExecutionEquivalence,
            WORKERS,
        ),
        obligation(
            "partitioned_volume_cdt",
            "A single connected body's volume mesh partitions across workers with owned and ghost regions, and the assembled mesh has the same validity and quality guarantees as the serial mesh and a stated speedup on at least 4 workers.",
            EvidenceKind::ExecutionEquivalence,
            DECOMP,
        ),
        obligation(
            "parallel_meshing_on_one_machine",
            "Meshing saturates the admitted CPU threads of one machine under the single thread plan, with identical output at 1, 2 and 4 threads.",
            EvidenceKind::ExecutionEquivalence,
            PARALLEL,
        ),
        obligation(
            "worker_loss_and_transfer",
            "Worker loss retries once with fencing, transfers are resumable and encrypted end to end, stale results never evict inputs, and durable supervisor restart and durable cancellation recover.",
            EvidenceKind::ProcessLifecycle,
            WORKERS,
        ),
        obligation(
            "determinism_across_layouts_and_platforms",
            "Canonical topology and digest are identical across worker counts, partition layouts and the supported platforms, on a matrix that includes macOS and Linux.",
            EvidenceKind::ExecutionEquivalence,
            WORKERS,
        ),
        obligation(
            "budgets_and_cancellation",
            "Element, memory, scratch, search, recursion and time budgets and cooperative cancellation are enforced inside every stage with typed failures.",
            EvidenceKind::ResourceAdmission,
            MESH,
        ),
        obligation(
            "bounded_publication_and_cold_materialization",
            "Mesh publication and cold materialization are bounded and compact at representative scale (millions of elements) and never exceed declared codec or cache caps.",
            EvidenceKind::PublicationIntegrity,
            MESH,
        ),
        obligation(
            "single_scheduler_and_browser_policy",
            "Meshing reuses the execution configuration with no second scheduler, and the browser either selects a native or remote mesher or fails explicitly, never meshing silently.",
            EvidenceKind::RejectionInvariant,
            PUBLIC,
        ),
        obligation(
            "performance_slo_and_governance",
            "A versioned performance and resource SLO covers the corpus parts, current reports are regenerated each release, and the meshing governance validators (corpus, conformance, differential, reliability, release evidence) pass.",
            EvidenceKind::PerformanceMeasurement,
            PUBLIC,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_MESH_GEOMETRY_001,
    FEA_MESH_DISCRETIZATION_001,
    FEA_MESH_REALPARTS_001,
    FEA_MESH_EXECUTION_001,
];
