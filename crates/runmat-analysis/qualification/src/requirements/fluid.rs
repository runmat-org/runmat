//! Standing incompressible Newtonian, laminar, single-phase fluid requirements.
//! Coupled heat/interface physics is a separate qualification responsibility.
//! Witness links declare assertion scope; they do not declare passing qualification.
use crate::requirement::*;
const STEADY_FORMULATION: Formulation = Formulation::IncompressibleLaminarSteady;
const TRANSIENT_FORMULATION: Formulation = Formulation::IncompressibleLaminarTransient;
const STEADY_KERNEL: &[Envelope] = &[envelope(
    STEADY_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::NumericalKernel,
)];
const TRANSIENT_KERNEL: &[Envelope] = &[envelope(
    TRANSIENT_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::NumericalKernel,
)];
const STEADY_NATIVE: &[Envelope] = &[envelope(
    STEADY_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::NativeProcess,
)];
const TRANSIENT_NATIVE: &[Envelope] = &[envelope(
    TRANSIENT_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::NativeProcess,
)];
const STEADY_PUBLIC: &[Envelope] = &[envelope(
    STEADY_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::PublicStudy,
)];
const TRANSIENT_PUBLIC: &[Envelope] = &[envelope(
    TRANSIENT_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::PublicStudy,
)];
const STEADY_DURABLE: &[Envelope] = &[envelope(
    STEADY_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::Durable,
)];
const TRANSIENT_DURABLE: &[Envelope] = &[envelope(
    TRANSIENT_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::Durable,
)];
const CORE: &[Envelope] = &[
    STEADY_KERNEL[0],
    TRANSIENT_KERNEL[0],
    STEADY_NATIVE[0],
    TRANSIENT_NATIVE[0],
    STEADY_PUBLIC[0],
    TRANSIENT_PUBLIC[0],
    STEADY_DURABLE[0],
    TRANSIENT_DURABLE[0],
];
const TRANSIENT: &[Envelope] = &[
    TRANSIENT_KERNEL[0],
    TRANSIENT_NATIVE[0],
    TRANSIENT_PUBLIC[0],
    TRANSIENT_DURABLE[0],
];
const PUBLIC: &[Envelope] = &[
    STEADY_PUBLIC[0],
    TRANSIENT_PUBLIC[0],
    STEADY_NATIVE[0],
    TRANSIENT_NATIVE[0],
    STEADY_DURABLE[0],
    TRANSIENT_DURABLE[0],
];
const ALL: &[Envelope] = &[
    STEADY_KERNEL[0],
    TRANSIENT_KERNEL[0],
    STEADY_NATIVE[0],
    TRANSIENT_NATIVE[0],
    STEADY_PUBLIC[0],
    TRANSIENT_PUBLIC[0],
    STEADY_DURABLE[0],
    TRANSIENT_DURABLE[0],
    envelope(
        STEADY_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        TRANSIENT_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        STEADY_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        STEADY_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        TRANSIENT_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        TRANSIENT_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        STEADY_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
    envelope(
        TRANSIENT_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const TOPOLOGY_001: Requirement = Requirement {
    id: "FEA-FLUID-TOPOLOGY-001",
    title: "Canonical conservative finite-volume projection",
    stages: &[
        Stage::Fluid
    ],
    acceptance: concat!(
"Flow uses complete canonical tetrahedral control-volume geometry and exact ",
"cell/face authority. Meshes outside validated geometry and quality limits ",
"are rejected; no channel reduction substitutes for the original domain.",
),
    owners: &[
        "crates/runmat-analysis/fea/src/topology/finite_volume.rs"
    ],
    dependencies: &[
    ],
    obligations: &[
        obligation(
            "inventory",
            concat!(
"Retain canonical cell identities, complete reciprocal cell-face incidence, ",
"owner/neighbour, oriented area vectors, cell volumes/centroids, centroid ",
"distances, interpolation weights, non-orthogonality/skewness metrics and ",
"exact original boundary provenance.",
),
            EvidenceKind::PhysicalBalance,
            CORE
        ),
        obligation(
            "geometry",
            concat!(
"Projection is deterministic under reordered source access and placement; ",
"oriented surfaces satisfy the divergence theorem and exact source/region ",
"ownership. Cache identity binds the original mesh and projection policy.",
),
            EvidenceKind::ExecutionEquivalence,
            CORE
        ),
        obligation(
            "quality",
            concat!(
"Admit supported tetrahedral order/geometry and governed ",
"non-orthogonality/skewness limits before work. Reject inversion, ",
"degeneracy, unsupported curvature, nonmanifold/missing/ambiguous boundaries ",
"or mutated source geometry; never silently reduce the model.",
),
            EvidenceKind::RejectionInvariant,
            CORE
        ),
        obligation(
            "budget",
            concat!(
"Count construction inventory and checked scratch/traversal work before ",
"allocation; exact ceilings, overflow and cancellation govern topology work.",
),
            EvidenceKind::ResourceAdmission,
            CORE
        ),
    ],
};
const EQUATIONS_001: Requirement = Requirement {
    id: "FEA-FLUID-EQUATIONS-001",
    title: "Original incompressible laminar mass and momentum equations",
    stages: &[
        Stage::Fluid
    ],
    acceptance: concat!(
"The supported initial flow envelope is incompressible, Newtonian, laminar ",
"and single phase. Steady and transient operators conserve original mass and ",
"momentum, use documented interpolation/stabilization and governed ",
"skew/non-orthogonal corrections, and solve pressure-velocity coupling with ",
"admissible shared nonsymmetric algebra.",
),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/momentum.rs",
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/pressure.rs",
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/iterate.rs"
    ],
    dependencies: &[
        concat!(
"Shared canonical topology, exact material assignment and general sparse ",
"solver/resource authorities.",
)
    ],
    obligations: &[
        obligation(
            "flux",
            concat!(
"Evaluate one original oriented owner-face mass flux and conservative ",
"advective, viscous and pressure momentum actions with equal/opposite ",
"internal-face contributions. Document interpolation, stabilization and ",
"non-orthogonal/skewness corrections within validated mesh limits.",
),
            EvidenceKind::PhysicalBalance,
            CORE
        ),
        obligation(
            "coupling",
            concat!(
"Pressure-velocity coupling solves original incompressibility and momentum ",
"equations. Component gauges/nullspaces are explicit, invalid or ",
"incompatible references reject, and physical residual checks cannot be ",
"replaced by solver convergence. Incompressibility is part of the mixed ",
"fluid formulation, never a generic user-authored equality constraint.",
),
            EvidenceKind::PhysicalBalance,
            CORE
        ),
        obligation(
            "pressure_state",
            concat!(
"Keep original reference-relative pressure as governing state through ",
"iteration, recovery and cold execution. Absolute pressure is a projection; ",
"converting it back must not erase fluctuations. Bind pressure references ",
"and retain owner-face flux rather than reconstructing missing original ",
"state.",
),
            EvidenceKind::OriginalStateComparison,
            CORE
        ),
    ],
};
const INPUTS_001: Requirement = Requirement {
    id: "FEA-FLUID-INPUTS-001",
    title: "Exact typed fluid boundaries, sources and initial conditions",
    stages: &[Stage::Fluid],
    acceptance: concat!(
        "Fluid cell domains, exact boundary regions, material laws, constraints, ",
        "sources and initial fields are typed original authority. Shared ",
        "spatial/time signal realization preserves declared flow and momentum ",
        "resultants and rejects wrong-dimensional, absent, ambiguous, conflicting or ",
        "incompatible bindings.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/boundary.rs",
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/initial.rs",
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/forcing.rs",
    ],
    dependencies: &[concat!(
        "Shared typed materials, cell-domain realization, constraints, excitations ",
        "and initial-condition authority.",
    )],
    obligations: &[
        obligation(
            "boundaries",
            concat!(
                "Implement component velocity, tangential moving-wall, symmetry, periodic ",
                "and pressure-reference constraints plus inlet velocity, outlet pressure and ",
                "mass-flow boundaries on exact regions. Reject unsupported normal wall ",
                "motion unless a qualified ALE formulation governs it.",
            ),
            EvidenceKind::SemanticRealization,
            CORE,
        ),
        obligation(
            "signals",
            concat!(
                "Use shared typed signals for inlet velocity, pressure, mass flow, wall ",
                "motion and body acceleration. Realize space/time dependence conservatively, ",
                "preserving declared mass/momentum resultants over whole boundaries, cells ",
                "and partitions with explicit sign/frame conventions and event sidedness.",
            ),
            EvidenceKind::SemanticRealization,
            CORE,
        ),
        obligation(
            "initial",
            concat!(
                "Resolve model-owned reusable initial velocity and pressure/reference fields ",
                "on exact cell scope from one typed finite uniform value, exact ",
                "non-frequency signal sample with coordinate/discontinuity side, or ",
                "immutable complete field artifact. Check content, topology, quantity, shape ",
                "and explicit vector frame. Study binding selects the consuming step; ",
                "compatible upstream state takes precedence unless a typed replacement ",
                "policy applies, and conflicts reject. Validate face flux and ",
                "constraint/gauge consistency before transient execution; no fabricated ",
                "terminal profile or inferred extremal boundary authority.",
            ),
            EvidenceKind::SemanticRealization,
            TRANSIENT,
        ),
        obligation(
            "rejection",
            concat!(
                "Reject missing/multiply matched/wrong-dimension material, cell, source or ",
                "boundary authority, quantity conflicts, stale input and exceeded input ",
                "realization budgets before numerical execution.",
            ),
            EvidenceKind::RejectionInvariant,
            CORE,
        ),
    ],
};
const TRANSIENT_001: Requirement = Requirement {
    id: "FEA-FLUID-TRANSIENT-001",
    title: "Physical transient integration and original cold chronology",
    stages: &[
        Stage::Fluid
    ],
    acceptance: concat!(
"Transient flow integrates original fluid inertia and pressure-velocity ",
"evolution over actual intervals. Immutable original cell velocity, pressure ",
"reference/fluctuation and owner-face flux govern cold continuation; ",
"schedules, events, work limits and outputs retain their original chronology.",
),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/transient.rs",
        "crates/runmat-analysis/fea/src/execution/fluid/time_block.rs"
    ],
    dependencies: &[
        concat!(
"Mandatory independent transient accuracy, admitted task resources and ",
"compatible complete state publication.",
)
    ],
    obligations: &[
        obligation(
            "inertia",
            concat!(
"Integrate actual momentum time derivatives rather than return a terminal ",
"steady profile; initial state and step interval bind the original inertial ",
"operator.",
),
            EvidenceKind::IndependentReference,
            TRANSIENT
        ),
        obligation(
            "convergence",
            concat!(
"Show temporal convergence independently of spatial error on a nontrivial ",
"transient reference, including a transient vortex; report field error and ",
"conservative balance.",
),
            EvidenceKind::TemporalConvergence,
            TRANSIENT
        ),
        obligation(
            "history",
            concat!(
"Preserve complete original interval state, scheduled observations, ",
"one-sided discontinuities, predecessor order and cumulative admitted work. ",
"Reject reordered/substituted state, invalid intervals, budgets and ",
"cancellation without accepting a suffix.",
),
            EvidenceKind::ColdReplay,
            TRANSIENT
        ),
        obligation(
            "producer_exit",
            concat!(
"After actual native producer termination a fresh consumer resumes original ",
"transient fields and outputs without producer memory; split and ",
"uninterrupted histories agree within the governed accuracy envelope.",
),
            EvidenceKind::ProcessLifecycle,
            TRANSIENT_NATIVE
        ),
    ],
};
const RECOVERY_001: Requirement = Requirement {
    id: "FEA-FLUID-RECOVERY-001",
    title: "Original fluid fields and physical conservation recovery",
    stages: &[
        Stage::Fluid
    ],
    acceptance: concat!(
"Recover original cell and face pressure, velocity, flux, stress and wall ",
"quantities consistently, retaining exact boundary ownership and signed ",
"mass/momentum balance. Acceptance includes original equation residual, ",
"continuity, constitutive face-flux closure and global/per-cell/per-boundary ",
"physical balance.",
),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/recovery.rs",
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/boundary_balances.rs"
    ],
    dependencies: &[
    ],
    obligations: &[
        obligation(
            "fields",
            concat!(
"Retain original cell fields and owner-face flux; recover face traces, ",
"stresses and wall quantities with declared location, units, frames and sign ",
"conventions. Cold consumers validate original source and reject changed ",
"fields/tractions rather than silently re-solving them.",
),
            EvidenceKind::OriginalStateComparison,
            CORE
        ),
        obligation(
            "balance",
            concat!(
"Report global, per-cell and per-boundary signed mass and momentum ",
"components from original forces/fluxes with physical normalization. ",
"Algebraic residual convergence alone cannot pass physical acceptance.",
),
            EvidenceKind::PhysicalBalance,
            CORE
        ),
    ],
};
const ACCURACY_001: Requirement = Requirement {
    id: "FEA-FLUID-ACCURACY-001",
    title: "Independent flow benchmark and convergence acceptance",
    stages: &[Stage::Fluid],
    acceptance: concat!(
        "Production flow requires independent Poiseuille, Couette, lid-driven ",
        "cavity, backward-facing-step references within the declared Reynolds range, ",
        "a transient vortex, spatial/time convergence and conservation. Every ",
        "mandatory accuracy criterion must be satisfied by applicable current ",
        "evidence before publication; a missing independent discretization bound ",
        "remains open.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/qualification.rs",
        "crates/runmat-analysis/fea/src/results/accuracy/evaluation.rs",
    ],
    dependencies: &[concat!(
        "References are independent of the operator under test; original meshes, ",
        "tolerances and caps remain mandatory.",
    )],
    obligations: &[
        obligation(
            "steady_references",
            concat!(
                "Validate Poiseuille and Couette plus lid-driven cavity and backward-facing ",
                "step across the documented Reynolds range, with velocity/pressure/flow/wall ",
                "quantities and conservative closure.",
            ),
            EvidenceKind::IndependentReference,
            STEADY_KERNEL,
        ),
        obligation(
            "spatial",
            concat!(
                "Demonstrate spatial convergence and absolute error for non-affine flow, ",
                "pressure, integrated flow and wall stress within the documented ",
                "quality/Reynolds envelope; affine exactness alone does not establish ",
                "convergence.",
            ),
            EvidenceKind::SpatialConvergence,
            STEADY_KERNEL,
        ),
        obligation(
            "acceptance",
            concat!(
                "Shared accuracy assessment consumes original result, receipts and ",
                "applicable independent spatial/time evidence. Missing, failed or ",
                "indeterminate mandatory criteria block completed semantic publication; ",
                "imported provenance cannot create numerical proof.",
            ),
            EvidenceKind::AccuracyAcceptance,
            PUBLIC,
        ),
    ],
};
const EXECUTION_001: Requirement = Requirement {
    id: "FEA-FLUID-EXECUTION-001",
    title: "Governed fluid execution and semantic publication",
    stages: &[Stage::Fluid],
    acceptance: concat!(
        "Fluid execution retains original equations and fields across supported ",
        "placement, admitted resources and immutable publication. Serial correctness ",
        "precedes partitioned cell assembly and distributed linear algebra. Every ",
        "remote, distributed or accelerator claim requires direct applicable ",
        "execution and equivalence evidence; host fallback or fanout is not that ",
        "evidence.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/execution/fluid/mod.rs",
        "crates/runmat-analysis/execution/src/fluid_study.rs",
        "crates/runmat-execution-runner-native/src/analysis_kernel/fluid_steady.rs",
    ],
    dependencies: &[
        concat!(
            "Shared immutable artifacts, task admission, state/observation/accuracy ",
            "authorities and attested release/documentation.",
        ),
        concat!(
            "Conjugate heat transfer requires separately qualified thermal and fluid ",
            "operators and conservative exact-interface closure; fluid evidence alone ",
            "cannot qualify the coupled system.",
        ),
    ],
    obligations: &[
        obligation(
            "resources",
            concat!(
                "Admit original projection, source/initial realization, pressure/momentum ",
                "operators, solver work, retained histories/outputs and transport before ",
                "execution; cancellation and exact ceilings reject safely. Measured ",
                "whole-process/backend usage must support claimed resource envelopes; ",
                "requested bytes are not RSS.",
            ),
            EvidenceKind::ResourceAdmission,
            ALL,
        ),
        obligation(
            "placement",
            concat!(
                "Implement partitioned cell assembly after verified serial equations, then ",
                "distributed linear algebra. Preserve original boundary/source flow and ",
                "momentum resultants and numerical fields across ",
                "serial/native/durable/encrypted-remote/partitioned/claimed-accelerator ",
                "execution with supported precision and authenticated lifecycle.",
            ),
            EvidenceKind::ExecutionEquivalence,
            ALL,
        ),
        obligation(
            "publication",
            concat!(
                "Publish original complete state, observations, receipts and mandatory ",
                "accuracy evidence atomically under exact source/phase lineage. Failed or ",
                "partial imports cannot install completed authority. Include ",
                "supported-envelope documentation and reproducible release evidence; no ",
                "implicit channel fallback.",
            ),
            EvidenceKind::PublicationIntegrity,
            PUBLIC,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    TOPOLOGY_001,
    EQUATIONS_001,
    INPUTS_001,
    TRANSIENT_001,
    RECOVERY_001,
    ACCURACY_001,
    EXECUTION_001,
];
