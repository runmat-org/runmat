//! Standing harmonic-acoustic acceptance. Numerical closure, continuum accuracy,
//! uniqueness, cold provenance and supported execution are distinct obligations.
use crate::requirement::*;
const KERNEL: &[Envelope] = &[
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
];
const NATIVE: &[Envelope] = &[
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet4,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet10,
        ExecutionEnvelope::NativeProcess,
    ),
];
const TOPOLOGY: &[Envelope] = &[NATIVE[0], NATIVE[1], PUBLIC[0], PUBLIC[1]];
const BRIDGE: &[Envelope] = &[
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet4,
        ExecutionEnvelope::PreparedController,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet10,
        ExecutionEnvelope::PreparedController,
    ),
];
const PUBLIC: &[Envelope] = &[
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
];
const EXECUTION: &[Envelope] = &[
    NATIVE[0],
    NATIVE[1],
    PUBLIC[0],
    PUBLIC[1],
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const CANONICAL: Requirement = Requirement {
    id: "FEA-ACOUSTIC-OPERATOR-001",
    title: "Canonical material-region Helmholtz operator",
    stages: &[Stage::Acoustics, Stage::SharedAuthorities],
    acceptance: concat!(
        "Assemble the pressure Helmholtz weak form with volume terms ",
        "integral(grad(q) dot grad(p)/rho) minus omega squared times ",
        "integral(q*p/(rho*c squared)), and consistent natural/Robin/source terms ",
        "under one explicit harmonic phase convention, on the canonical volume and ",
        "exact boundary mesh, using each physical region's density and sound speed, ",
        "compatible complex coefficients and deterministic shared sparse ",
        "contributions. Preserve original topology, entity ordering and integration ",
        "authority through execution. Generated rectangular graphs, inferred ",
        "geometry and substitute solver topology are forbidden in production.",
    ),
    owners: &["crates/runmat-analysis/fea/src/physics/acoustic/canonical/assembly.rs"],
    dependencies: &[concat!(
        "Shared topology, properties, units, integration, sparse assembly and solver ",
        "contracts remain authoritative.",
    )],
    obligations: &[
        obligation(
            "integrals",
            concat!(
                "Independent exact simplex or manufactured references verify volume ",
                "gradient/mass and boundary terms for both element orders, orientations and ",
                "physical material scaling.",
            ),
            EvidenceKind::IndependentReference,
            KERNEL,
        ),
        obligation(
            "regions",
            concat!(
                "Material-interface pressure, normal velocity and power continuity agree ",
                "with independent references under spatial refinement on exact assigned ",
                "regions.",
            ),
            EvidenceKind::SpatialConvergence,
            KERNEL,
        ),
        obligation(
            "topology",
            concat!(
                "Public execution consumes the authored canonical mesh without generated ",
                "solver-graph substitution, and retains original field topology through ",
                "native recovery and cold transport.",
            ),
            EvidenceKind::SemanticRealization,
            TOPOLOGY,
        ),
    ],
};
const BOUNDARIES: Requirement = Requirement {
    id: "FEA-ACOUSTIC-BOUNDARY-001",
    title: "Exact typed acoustic boundaries and complex constraints",
    stages: &[Stage::Acoustics],
    acceptance: concat!(
        "Implement prescribed pressure, normal velocity, rigid wall, impedance and ",
        "supported radiation boundaries on exact resolved regions, and typed ",
        "periodic/Bloch-Floquet relations in their declared envelope. Compile ",
        "equality constraints through shared authority; natural/Robin boundaries ",
        "remain operator/source terms rather than constrained booleans. Reject ",
        "unsupported, ambiguous, conflicting or overlapping conditions before solve.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/acoustic/canonical/boundary.rs",
        "crates/runmat-analysis/fea/src/physics/acoustic/canonical/periodic.rs",
    ],
    dependencies: &[concat!(
        "Shared constraint reduction and excitation realization preserve phase, ",
        "units, frames and region identities.",
    )],
    obligations: &[
        obligation(
            "lifting",
            concat!(
                "Complex prescribed pressure lifting agrees with an independently assembled r",
                "educed block system.",
            ),
            EvidenceKind::IndependentReference,
            KERNEL,
        ),
        obligation(
            "phase",
            concat!(
                "Periodic and Bloch/Floquet constraints preserve phase and paired traces, ",
                "pressure, velocity and signed power against independent references under ",
                "refinement.",
            ),
            EvidenceKind::SpatialConvergence,
            KERNEL,
        ),
        obligation(
            "radiation",
            concat!(
                "Supported radiation operator and recovered pressure, phase, mean velocity ",
                "and signed power satisfy independent references and spatial convergence on ",
                "their declared geometry/frequency envelope.",
            ),
            EvidenceKind::SpatialConvergence,
            KERNEL,
        ),
        obligation(
            "reject",
            concat!(
                "Reject unmatched periodic pairs, unsupported radiation/incident regions, ",
                "conflicting prescribed pressure and overlapping sources/natural boundaries ",
                "without guessing a target or constraint.",
            ),
            EvidenceKind::RejectionInvariant,
            KERNEL,
        ),
    ],
};
const SOURCES: Requirement = Requirement {
    id: "FEA-ACOUSTIC-SOURCE-001",
    title: "Shared complex source spectra with conserved realization",
    stages: &[Stage::Acoustics, Stage::Authoring],
    acceptance: concat!(
        "Realize complex pressure, normal/total volume velocity and supported ",
        "incident-field spectra through shared typed signals and excitations. Freeze ",
        "callable generation into bounded immutable preprocessing artifacts before ",
        "assembly. Preserve canonical units, complex phase convention, domain ",
        "coverage, target, frame, distribution and resultant normalization without ",
        "family-private amplitude/phase controls or live solver callbacks. Explicit ",
        "and generated spectra must agree in amplitude, phase and boundary power.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/excitation/mod.rs",
        "crates/runmat-analysis/fea/src/physics/acoustic/canonical/boundary.rs",
    ],
    dependencies: &[concat!(
        "Shared signal/generator freezing and typed authoring requirements apply to ",
        "all supported acoustic sources.",
    )],
    obligations: &[
        obligation(
            "resultant",
            concat!(
                "Total volume flow preserves the authored complex patch resultant ",
                "independent of area or refinement and agrees with equivalent normal ",
                "velocity without duplicating that resultant.",
            ),
            EvidenceKind::PhysicalBalance,
            KERNEL,
        ),
        obligation(
            "incident",
            concat!(
                "Incident/reflected/transmitted pressure and net power converge to ",
                "independent references; incoming power is not counted twice at small ",
                "amplitude.",
            ),
            EvidenceKind::SpatialConvergence,
            KERNEL,
        ),
        obligation(
            "spectra",
            concat!(
                "Explicit and frozen generated pressure/normal-velocity/volume-flow/incident ",
                "spectra produce equivalent original complex fields and boundary power ",
                "across the authored frequency inventory, including cold publication.",
            ),
            EvidenceKind::ExecutionEquivalence,
            PUBLIC,
        ),
    ],
};
const NUMERICS: Requirement = Requirement {
    id: "FEA-ACOUSTIC-NUMERICS-001",
    title: "Independent acoustic fields, resonance and physical balances",
    stages: &[Stage::Acoustics],
    acceptance: concat!(
        "Solve complex Helmholtz systems using an appropriate shared ",
        "indefinite/general solver and explicit resonance diagnostics. Recover ",
        "pressure, particle velocity, intensity, energy and boundary power ",
        "consistently from original accepted fields. Qualify ducts, rigid cavities, ",
        "impedance terminations, radiation, modal frequencies and spatial ",
        "convergence with independent references. Residual or passive power closure ",
        "alone cannot establish uniqueness or continuum accuracy.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/acoustic/canonical/response.rs",
        "crates/runmat-analysis/fea/src/solve/certificate.rs",
    ],
    dependencies: &[concat!(
        "Radiation convergence remains a mandatory boundary obligation; candidate ",
        "inversion success is not conditioning certification.",
    )],
    obligations: &[
        obligation(
            "duct",
            concat!(
                "Independent duct/impedance traveling waves verify complex pressure/velocity ",
                "phase, intensity, energy and boundary power under mesh refinement.",
            ),
            EvidenceKind::SpatialConvergence,
            KERNEL,
        ),
        obligation(
            "cavity",
            concat!(
                "Rigid cavity references verify modal/resonance frequencies, fields and ",
                "phase reversal on both sides of poles under refinement; pole response is ",
                "not accepted solely because its physical right-hand side is zero.",
            ),
            EvidenceKind::IndependentReference,
            KERNEL,
        ),
        obligation(
            "conditioning",
            concat!(
                "Report source-independent resonance conditioning or sensitivity diagnostics ",
                "over the supported frequency envelope and independently verify modal ",
                "frequencies. A sufficient nonsingularity certificate does not establish ",
                "good conditioning, bounded forward error or complete modal-spectrum ",
                "accuracy.",
            ),
            EvidenceKind::IndependentReference,
            KERNEL,
        ),
        obligation(
            "uniqueness",
            concat!(
                "The sparse LU that solves the original reduced operator certifies its ",
                "nonsingularity (estimated right-inverse defect contraction below the ",
                "authored limit) before physical residual/power closure is accepted; above ",
                "the limit the solve fails closed. No artificial damping or sampled small ",
                "matrix substitutes for the operator.",
            ),
            EvidenceKind::IndependentReference,
            KERNEL,
        ),
        obligation(
            "cold_claim",
            concat!(
                "Retained factorization certificates bind the authored limit, original ",
                "operator identity, result and numerical receipt. Cold hashes or decoded ",
                "producer claims never become trusted mathematical proof; independent ",
                "replay requires refactoring the original operator.",
            ),
            EvidenceKind::OriginalStateComparison,
            KERNEL,
        ),
    ],
};
const ACCURACY: Requirement = Requirement {
    id: "FEA-ACOUSTIC-ACCURACY-001",
    title: "Mandatory acoustic accuracy gates publication",
    stages: &[Stage::Acoustics, Stage::ExecutionContracts],
    acceptance: concat!(
        "Every acoustic entry retains independently authenticated ",
        "original numerical evidence and its authored accuracy intent. Evaluate ",
        "mandatory field/convergence, physical balance, source realization and ",
        "requested resonance/uniqueness requirements without weakening tolerances or ",
        "inferring scientific adequacy from solver success. Missing, failed or ",
        "indeterminate mandatory accuracy blocks dependent state and ",
        "semantic publication.",
    ),
    owners: &[
        "crates/runmat-analysis/execution/src/acoustic_study.rs",
        "crates/runmat-analysis/fea/src/execution/acoustic/receipt.rs",
    ],
    dependencies: &[concat!(
        "Shared accuracy contracts and semantic publication preserve exact numerical ",
        "origin and accepted formulation envelope.",
    )],
    obligations: &[
        obligation(
            "lineage",
            concat!(
                "Accuracy preparation and receipt validation require the exact original ",
                "workload, mesh, result, authored intent and numerical receipt, rejecting ",
                "missing, duplicate, foreign or weakened contexts.",
            ),
            EvidenceKind::AccuracyAcceptance,
            BRIDGE,
        ),
        obligation(
            "gate",
            concat!(
                "Public completion and dependent publication reject missing or failed ",
                "mandatory acoustic accuracy, and publish indeterminate accuracy only under ",
                "an explicit markIndeterminate policy; release qualification is not ",
                "inferred from an accepted result alone.",
            ),
            EvidenceKind::AccuracyAcceptance,
            PUBLIC,
        ),
    ],
};
const EXECUTION_REQUIREMENT: Requirement = Requirement {
    id: "FEA-ACOUSTIC-EXECUTION-001",
    title: "Governed acoustic frequency execution and original-field publication",
    stages: &[Stage::Acoustics, Stage::ExecutionContracts, Stage::Release],
    acceptance: concat!(
        "Parallelize independent frequencies through shared admitted task ",
        "orchestration and deterministically compose the exact ordered authored ",
        "inventory. Preserve each entry's original complex state, numerical receipt, ",
        "accuracy decision and source fields through immutable publication and cold ",
        "bounded access. Admit complete input/operator/uniqueness/output/transport ",
        "lifetimes and work before execution, retain real accounting and measured ",
        "resource evidence, and qualify each advertised local, durable, encrypted ",
        "remote, partitioned or accelerator envelope. Frequency fanout is not ",
        "distributed linear algebra and host fallback is not accelerator execution. ",
        "Publish supported envelopes, documentation, examples and release evidence.",
    ),
    owners: &[
        "crates/runmat-analysis/execution/src/acoustic_frequency_assembly.rs",
        "crates/runmat-execution-runner-native/src/analysis_kernel/acoustic_frequency_assembly.rs",
        "crates/runmat-runtime/src/analysis/acoustic_study_publication.rs",
    ],
    dependencies: &[concat!(
        "Shared execution, immutable objects, authenticated completion, bounded ",
        "readers and release gates remain mandatory.",
    )],
    obligations: &[
        obligation(
            "inventory",
            concat!(
                "Exact frequency fan-in preserves all original entries, roots, ",
                "pressure/velocity fields and metadata independent of completion order or ",
                "chunking, rejecting missing, duplicate or substituted entries.",
            ),
            EvidenceKind::PublicationIntegrity,
            EXECUTION,
        ),
        obligation(
            "selection",
            concat!(
                "Cold full/partial complex reads use exact sample and original partitions ",
                "without solver reruns, nearest-frequency selection or ",
                "aggregate-as-single-state substitution; corruption and incomplete roots ",
                "reject.",
            ),
            EvidenceKind::PublicationIntegrity,
            EXECUTION,
        ),
        obligation(
            "admission",
            concat!(
                "Worker rederivation rejects altered workload/uniqueness controls or ",
                "understated storage/work plans before work; original authored ceilings ",
                "remain enforced. Whole-task measured peak usage, cancellation and terminal ",
                "accounting match claimed envelopes.",
            ),
            EvidenceKind::ResourceAdmission,
            EXECUTION,
        ),
        obligation(
            "placement",
            concat!(
                "Every claimed execution placement has actual end-to-end equivalence, ",
                "failure/cancellation and resource evidence. Durable producer termination, ",
                "encrypted remote and partitioned/GPU behavior receive separate proof rather ",
                "than local transport/fanout credit.",
            ),
            EvidenceKind::ExecutionEquivalence,
            EXECUTION,
        ),
        obligation(
            "release",
            concat!(
                "Release evidence and usable examples document supported boundaries, ",
                "sources, material/mesh/frequency limits, diagnostics and accuracy; all ",
                "blocking release criteria pass before advertisement.",
            ),
            EvidenceKind::AccuracyAcceptance,
            EXECUTION,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    CANONICAL,
    BOUNDARIES,
    SOURCES,
    NUMERICS,
    ACCURACY,
    EXECUTION_REQUIREMENT,
];
