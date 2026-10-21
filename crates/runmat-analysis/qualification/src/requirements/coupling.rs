//! Standing physical acceptance requirements. Witness links describe assertion scope,
//! not successful execution or release qualification.
use crate::requirement::*;
const THERMOELASTIC: &[Envelope] = &[
    envelope(
        Formulation::Thermoelastic,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::Thermoelastic,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
];
const THERMOELASTIC_PUBLIC: &[Envelope] = &[
    envelope(
        Formulation::Thermoelastic,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::Thermoelastic,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
];
const ELECTROTHERMAL_PUBLIC: &[Envelope] = &[
    envelope(
        Formulation::Electrothermal,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::Electrothermal,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
];
const CHT: &[Envelope] = &[envelope(
    Formulation::ConjugateHeatTransfer,
    MeshOrder::CanonicalMixedTopology,
    ExecutionEnvelope::PublicStudy,
)];
const FSI: &[Envelope] = &[envelope(
    Formulation::FluidStructureInteraction,
    MeshOrder::CanonicalMixedTopology,
    ExecutionEnvelope::PublicStudy,
)];
const COUPLED: &[Envelope] = &[
    envelope(
        Formulation::Thermoelastic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::Electrothermal,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::ConjugateHeatTransfer,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::FluidStructureInteraction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PublicStudy,
    ),
];
const FSI_GOVERNED: &[Envelope] = &[
    FSI[0],
    envelope(
        Formulation::FluidStructureInteraction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::FluidStructureInteraction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::FluidStructureInteraction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::FluidStructureInteraction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const FEA_COUPLING_INTERFACE_001: Requirement = Requirement {
    id: "FEA-COUPLING-INTERFACE-001",
    title: "Typed physical interfaces and conservative transfers",
    stages: &[
        Stage::Thermal,
        Stage::Electromagnetics,
        Stage::Fluid,
        Stage::FluidStructureInteraction,
    ],
    acceptance: concat!(
        "Coupled procedures retain each domain equation, exact admitted geometry ",
        "identity and revision, distinct conservation law and explicit physical ",
        "interface law. Numerical enforcement and relaxation belong to typed ",
        "procedure/solver policy.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/constraints/coupling.rs",
        "crates/runmat-analysis/fea/src/physics/fsi/interface.rs",
    ],
    dependencies: &[],
    obligations: &[
        obligation(
            "identity",
            concat!(
                "Resolve exact interface sides with original geometry/revision and ",
                "deterministic mapping; CHT declares exactly one perfect-transfer, ",
                "conductance or resistance law; FSI declares physical kinematics. Do not ",
                "infer terminals or interfaces from coordinate extrema.",
            ),
            EvidenceKind::SemanticRealization,
            COUPLED,
        ),
        obligation(
            "transfer",
            concat!(
                "Validate matching or nonmatching transfer maps, conservation, error, ",
                "frame/orientation, units, source/target lineage and state compatibility ",
                "before consumption. Preserve each side conservation law and original ",
                "interface evidence.",
            ),
            EvidenceKind::PhysicalBalance,
            COUPLED,
        ),
        obligation(
            "seams",
            concat!(
                "Two-way policies, executable constitutive laws and future ",
                "derivative/optimization capability remain explicitly bounded; capability ",
                "metadata cannot imply implemented coupling, gradients or user-law callbacks ",
                "inside assembly, recovery, iterative solves or distributed ranks.",
            ),
            EvidenceKind::RejectionInvariant,
            COUPLED,
        ),
    ],
};
const FEA_COUPLING_THERMOELASTIC_001: Requirement = Requirement {
    id: "FEA-COUPLING-THERMOELASTIC-001",
    title: "Solved-temperature thermoelastic transitions and closure",
    stages: &[Stage::Thermal],
    acceptance: concat!(
        "One-way thermoelasticity consumes a solved thermal field on the same mesh ",
        "or through a validated map as an explicit typed sequence transition. ",
        "Temperature reference, projection, thermal strain and complete state ",
        "lineage remain attributable.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/pipeline/thermo_mechanical/solved.rs",
        "crates/runmat-analysis/fea/src/assembly/thermoelastic_load.rs",
    ],
    dependencies: &[],
    obligations: &[
        obligation(
            "strain",
            concat!(
                "Evaluate temperature-dependent properties and thermal strain at integration ",
                "points using the declared reference temperature and original thermal field; ",
                "retain transfer error and mechanical equilibrium with thermal balance.",
            ),
            EvidenceKind::IndependentReference,
            THERMOELASTIC,
        ),
        obligation(
            "transition",
            concat!(
                "Thermal preload may feed only supported prestressed modal, harmonic and ",
                "buckling branches after complete physical state, constraints, tangent, ",
                "transition and mandatory accuracy evidence pass. An initial guess or ",
                "incomplete checkpoint is not a base state.",
            ),
            EvidenceKind::OriginalStateComparison,
            THERMOELASTIC_PUBLIC,
        ),
        obligation(
            "two_way",
            concat!(
                "Name one-way and two-way policies explicitly. Implement two-way ",
                "deformation-dependent thermal exchange only inside its supported envelope ",
                "and iterate to field and energy closure; reject unsupported feedback.",
            ),
            EvidenceKind::PhysicalBalance,
            THERMOELASTIC_PUBLIC,
        ),
    ],
};
const FEA_COUPLING_ELECTROTHERMAL_001: Requirement = Requirement {
    id: "FEA-COUPLING-ELECTROTHERMAL-001",
    title: "Original Joule-heat transfer and electrical/thermal iteration",
    stages: &[Stage::Electromagnetics, Stage::Thermal],
    acceptance: concat!(
        "Compute Joule heat from solved conduction or electromagnetic fields at ",
        "elements/integration points and feed the real thermal operator. Resolve ",
        "authored terminals; iterate temperature-dependent conductivity through a ",
        "documented policy to electrical residual, thermal residual, field-change ",
        "and energy closure.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/execution/electrothermal.rs",
        "crates/runmat-analysis/fea/src/execution/electrothermal/transfer.rs",
    ],
    dependencies: &[],
    obligations: &[
        obligation(
            "heat",
            concat!(
                "Transfer original solved-field Joule density and total power with exact ",
                "point/element topology, units and authenticated source/target lineage; ",
                "terminal electrical power and supplied thermal power agree.",
            ),
            EvidenceKind::PhysicalBalance,
            ELECTROTHERMAL_PUBLIC,
        ),
        obligation(
            "feedback",
            concat!(
                "Update temperature-dependent conductivity and re-solve the actual ",
                "electrical and thermal equations under documented iteration controls; ",
                "retain electrical residual, thermal residual, field change and energy ",
                "closure for every accepted coupled state.",
            ),
            EvidenceKind::AccuracyAcceptance,
            ELECTROTHERMAL_PUBLIC,
        ),
        obligation(
            "cold",
            concat!(
                "Cold transfer authenticates the complete original electrical result and ",
                "intended thermal consumer before use; reject foreign lineage, altered ",
                "power/inventory, unsupported equations, feedback and production intent.",
            ),
            EvidenceKind::ColdReplay,
            ELECTROTHERMAL_PUBLIC,
        ),
    ],
};
const FEA_COUPLING_CHT_001: Requirement = Requirement {
    id: "FEA-COUPLING-CHT-001",
    title: "Production thermal/fluid conjugate heat transfer",
    stages: &[Stage::Fluid, Stage::Thermal],
    acceptance: concat!(
        "CHT requires independently production-ready thermal and fluid operators and ",
        "exact solid/fluid interfaces. Re-solve both domains until interface and ",
        "global energy closure pass rather than reporting an inferred terminal heat ",
        "profile.",
    ),
    owners: &["crates/runmat-analysis/fea/src/physics/coupling/mod.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "interface",
            concat!(
                "Conservatively transfer temperature and heat flux on exact solid/fluid ",
                "sides under documented perfect-transfer, conductance or contact-resistance ",
                "law and numerical relaxation policy.",
            ),
            EvidenceKind::PhysicalBalance,
            CHT,
        ),
        obligation(
            "iteration",
            concat!(
                "Re-solve both physical domains to interface and global energy closure, ",
                "retaining per-iteration interface evidence, failure/cutback chronology and ",
                "validated spatial/temporal/coupling convergence.",
            ),
            EvidenceKind::AccuracyAcceptance,
            CHT,
        ),
        obligation(
            "dependencies",
            concat!(
                "Reject production CHT while either standalone operator, transfer, ",
                "state/resource admission or mandatory accuracy is unqualified; shared ",
                "signal/source and boundary energy attribution remain intact.",
            ),
            EvidenceKind::RejectionInvariant,
            CHT,
        ),
    ],
};
const FEA_COUPLING_FSI_001: Requirement = Requirement {
    id: "FEA-COUPLING-FSI-001",
    title: "Production dynamics/fluid interaction with stable conservative ALE execution",
    stages: &[Stage::FluidStructureInteraction],
    acceptance: concat!(
        "FSI depends on independently production-ready structural dynamics and CFD. ",
        "Exact interface mapping conserves traction, velocity and work while a ",
        "documented mesh-motion/ALE and partitioned iteration policy governs moving ",
        "domains.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/fsi/mod.rs",
        "crates/runmat-analysis/fea/src/physics/fsi/interface.rs",
    ],
    dependencies: &[],
    obligations: &[
        obligation(
            "interface",
            concat!(
                "Resolve exact interface sides and conservatively transfer traction and ",
                "velocity with interface action/reaction and work balances; original ",
                "structural and fluid equations must actually be re-solved.",
            ),
            EvidenceKind::PhysicalBalance,
            FSI,
        ),
        obligation(
            "motion",
            concat!(
                "Document supported mesh-motion/ALE and quality-triggered remeshing, ",
                "validate transferred temporal/interface state and reject unsupported large ",
                "motion or topology change explicitly.",
            ),
            EvidenceKind::OriginalStateComparison,
            FSI,
        ),
        obligation(
            "iteration",
            concat!(
                "Use documented stable partitioned iteration, relaxation and added-mass ",
                "stabilization; qualify spatial, temporal and coupling-iteration convergence ",
                "and added-mass-sensitive independent reference cases.",
            ),
            EvidenceKind::IndependentReference,
            FSI,
        ),
        obligation(
            "orchestration",
            concat!(
                "Checkpoint distributed coupling orchestration with whole-task resource ",
                "admission, cancellation, bounded retry, interface history and publication ",
                "fencing; release evidence and documentation describe the actually qualified ",
                "envelope.",
            ),
            EvidenceKind::ProcessLifecycle,
            FSI_GOVERNED,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_COUPLING_INTERFACE_001,
    FEA_COUPLING_THERMOELASTIC_001,
    FEA_COUPLING_ELECTROTHERMAL_001,
    FEA_COUPLING_CHT_001,
    FEA_COUPLING_FSI_001,
];
