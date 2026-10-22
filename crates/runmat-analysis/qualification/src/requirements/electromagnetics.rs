//! Standing physical acceptance requirements. Witness links describe assertion scope,
//! not successful execution or release qualification.
use crate::requirement::*;
const STATIC: &[Envelope] = &[
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
];
const MAGNETIC: &[Envelope] = &[
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
];
const MAXWELL: &[Envelope] = &[envelope(
    Formulation::HarmonicMaxwell,
    MeshOrder::EdgeElements,
    ExecutionEnvelope::NumericalKernel,
)];
const MAGNETIC_AND_MAXWELL: &[Envelope] = &[
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::EdgeElements,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
];
const ALL_KERNEL: &[Envelope] = &[
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::EdgeElements,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
];
const STATIC_PUBLIC: &[Envelope] = &[
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
];
const MAGNETIC_PUBLIC: &[Envelope] = &[
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
];
const ALL_EXECUTION: &[Envelope] = &[
    STATIC_PUBLIC[0],
    STATIC_PUBLIC[1],
    STATIC_PUBLIC[2],
    STATIC_PUBLIC[3],
    MAGNETIC_PUBLIC[0],
    MAGNETIC_PUBLIC[1],
    envelope(
        Formulation::Electrostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PublicStudy,
    ),
];
const FEA_EM_SCALAR_001: Requirement = Requirement {
    id: "FEA-EM-SCALAR-001",
    title: "Distinct electrostatic and conducting scalar-potential operators",
    stages: &[Stage::Electromagnetics],
    acceptance: concat!(
        "Electrostatics and steady electrical conduction are separately typed ",
        "physical formulations with their own constitutive response, source units ",
        "and acceptance; neither may be silently substituted for the other.",
    ),
    owners: &["crates/runmat-analysis/fea/src/physics/electrical/mod.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "operator",
            concat!(
                "Assemble dielectric electrostatic and conductivity steady-current potential ",
                "operators by physical material region on canonical topology. Resolve exact ",
                "terminals, sources and boundary regions; reject absent or incompatible ",
                "material and domain intent.",
            ),
            EvidenceKind::IndependentReference,
            STATIC,
        ),
        obligation(
            "recovery",
            concat!(
                "Recover electric field, displacement/charge, current density, loss, stored ",
                "energy and terminal quantities appropriate to the chosen equation, with ",
                "physical units, topology and original solved-potential lineage.",
            ),
            EvidenceKind::PhysicalBalance,
            STATIC,
        ),
    ],
};
const FEA_EM_INPUT_001: Requirement = Requirement {
    id: "FEA-EM-INPUT-001",
    title: "Typed electrical and magnetic sources and rank-correct constraints",
    stages: &[Stage::Electromagnetics],
    acceptance: concat!(
        "Use shared typed excitation and constraint authorities for authored ",
        "voltage, charge, current, current-density, complex phasor, incident and ",
        "supported port inputs. Constraint plans remain rank-correct and ",
        "attributable to authored objects.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/electrical/boundary.rs",
        "crates/runmat-analysis/fea/src/physics/magnetic/plan.rs",
    ],
    dependencies: &[],
    obligations: &[
        obligation(
            "scalar_constraints",
            concat!(
                "Support prescribed potential, equipotential terminals, floating conductors, ",
                "integral charge/current conditions and gauge/reference handling; diagnose ",
                "rank deficiency, conflicts, unsupported combinations and incorrect source ",
                "units before solving.",
            ),
            EvidenceKind::SemanticRealization,
            STATIC,
        ),
        obligation(
            "magnetic_constraints",
            concat!(
                "Declare gauge, symmetry, periodic and supported complex phase-shifted ",
                "relations for magnetic and Maxwell operators, with source compatibility and ",
                "rank evidence.",
            ),
            EvidenceKind::SemanticRealization,
            MAGNETIC_AND_MAXWELL,
        ),
        obligation(
            "excitation",
            concat!(
                "Shared signals and realized artifacts preserve ",
                "voltage/charge/current/current-density and phasor magnitude, phase, exact ",
                "region realization, parameter histories where admissible and source/power ",
                "balances; no family-private history or callback authority.",
            ),
            EvidenceKind::ExecutionEquivalence,
            ALL_KERNEL,
        ),
    ],
};
const FEA_EM_MAGNETOSTATIC_001: Requirement = Requirement {
    id: "FEA-EM-MAGNETOSTATIC-001",
    title: "Documented magnetostatic formulation and physical recovery",
    stages: &[Stage::Electromagnetics],
    acceptance: concat!(
        "Declare vector or scalar potential, gauge, source representation and ",
        "validated material envelope. Resolve coils, currents, magnetic regions and ",
        "boundaries through canonical topology. Recover field, flux density, energy ",
        "and force only within independently supported formulation scope.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/magnetic/assembly.rs",
        "crates/runmat-analysis/fea/src/physics/magnetic/response.rs",
    ],
    dependencies: &[],
    obligations: &[
        obligation(
            "formulation",
            concat!(
                "Integrated magnetic operator and gauge enforce the documented physical ",
                "equations and source compatibility, with exact edge/region identities and ",
                "validated material restrictions.",
            ),
            EvidenceKind::IndependentReference,
            MAGNETIC,
        ),
        obligation(
            "fields",
            concat!(
                "Recover magnetic field, flux density and energy from the solved state, ",
                "report physical residual, gauge residual, source work and divergence/flux ",
                "closure; magnetic force needs separate formulation and accuracy evidence.",
            ),
            EvidenceKind::PhysicalBalance,
            MAGNETIC,
        ),
    ],
};
const FEA_EM_MAXWELL_001: Requirement = Requirement {
    id: "FEA-EM-MAXWELL-001",
    title: "Integrated harmonic edge-element Maxwell analysis",
    stages: &[Stage::Electromagnetics],
    acceptance: concat!(
        "Harmonic Maxwell is a distinct complex frequency-domain formulation using ",
        "canonical tetrahedral edge topology and integrated Nedelec functions rather ",
        "than approximate vector proxies.",
    ),
    owners: &["crates/runmat-analysis/fea/src/physics/magnetic/assembly.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "basis",
            concat!(
                "Complete oriented tetrahedral edge topology, integrated Nedelec basis and ",
                "material-interface continuity; assemble physical curl-curl, mass/loss, ",
                "source, boundary and gauge terms per material region.",
            ),
            EvidenceKind::IndependentReference,
            MAXWELL,
        ),
        obligation(
            "frequency",
            concat!(
                "Support complex coefficients and deterministic frequency-sweep composition, ",
                "complex source phasors, incident and supported port excitation with ",
                "reusable pattern/value identities and documented frequency conventions.",
            ),
            EvidenceKind::ExecutionEquivalence,
            MAXWELL,
        ),
        obligation(
            "reconstruction",
            concat!(
                "Use documented projection/reconstruction for vector recovery; retain ",
                "original divergence, energy, power, boundary and linear-solver balance ",
                "diagnostics.",
            ),
            EvidenceKind::PhysicalBalance,
            MAXWELL,
        ),
    ],
};
const FEA_EM_QUALIFICATION_001: Requirement = Requirement {
    id: "FEA-EM-QUALIFICATION-001",
    title: "Independent per-formulation evidence and governed electromagnetic execution",
    stages: &[Stage::Electromagnetics],
    acceptance: concat!(
        "Promote each electromagnetic formulation separately only after independent ",
        "physical, constraint, convergence, recovery, source and execution evidence ",
        "passes. Capability metadata and successful linear solves alone do not ",
        "establish release qualification.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/execution/electrical_static.rs",
        "crates/runmat-analysis/fea/src/execution/magnetic_static.rs",
    ],
    dependencies: &[],
    obligations: &[
        obligation(
            "references",
            concat!(
                "Pass independent capacitors, conductive bars, magnetostatic references, ",
                "cavity and waveguide modes, skin effect, material interfaces and ",
                "manufactured fields applicable to each formulation, with mesh and frequency ",
                "convergence and gauge/divergence, field, energy and power evidence.",
            ),
            EvidenceKind::SpatialConvergence,
            ALL_KERNEL,
        ),
        obligation(
            "execution",
            concat!(
                "Parallel independent frequencies and sources and justified mature operator ",
                "partitions preserve equivalent original fields and balances. Whole-task ",
                "resource admission, cancellation, retries, durable/remote lineage and ",
                "publication remain governed across every claimed placement.",
            ),
            EvidenceKind::ExecutionEquivalence,
            ALL_EXECUTION,
        ),
        obligation(
            "accuracy",
            concat!(
                "Publish correct formulation and release documentation with original ",
                "controls and evidence. Missing, failed or indeterminate mandatory accuracy ",
                "prevents publication; cold claims require authenticated original ",
                "source lineage and never reconstruct lost physics by decoding summaries.",
            ),
            EvidenceKind::PublicationIntegrity,
            ALL_EXECUTION,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_EM_SCALAR_001,
    FEA_EM_INPUT_001,
    FEA_EM_MAGNETOSTATIC_001,
    FEA_EM_MAXWELL_001,
    FEA_EM_QUALIFICATION_001,
];
