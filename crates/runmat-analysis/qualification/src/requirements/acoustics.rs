//! Standing harmonic-acoustic acceptance. Numerical closure, continuum accuracy,
//! uniqueness, cold provenance and supported execution are distinct obligations.
use crate::requirement::*;
const KERNEL: &[Envelope] = &[
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
];
const NATIVE: &[Envelope] = &[
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
];
const TOPOLOGY: &[Envelope] = &[NATIVE[0], NATIVE[1], PUBLIC[0], PUBLIC[1]];
const BRIDGE: &[Envelope] = &[
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuPreparedController,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuPreparedController,
    ),
];
const PUBLIC: &[Envelope] = &[
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuPublicStudy,
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
        ExecutionEnvelope::EncryptedRemote,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::PartitionedCpu,
    ),
    envelope(
        Formulation::AcousticHarmonic,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
macro_rules! witness {
    (
        $id:expr,
        $package:expr,
        $name:expr,
        $source:expr,
        $obligation:expr,
        $envelopes:expr,
        $boundary:expr
    ) => {
        Witness {
            id: $id,
            locator: WitnessLocator::CargoTest {
                package: $package,
                target: CargoTarget::Library,
                name: $name,
                source: $source,
            },
            claims: &[CoverageClaim {
                obligation: $obligation,
                envelopes: $envelopes,
                boundary: $boundary,
            }],
        }
    };
}
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
    witnesses: &[
        witness!(
            "FEA-WITNESS-ACOUSTIC-INTEGRALS",
            "runmat-analysis-fea",
            concat!(
                "physics::acoustic::canonical::tests::radiation::exact_operators::tet10_and_t",
                "ri6_all_entries_match_independent_simplex_moments_on_every_face",
            ),
            concat!(
                "crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/radiation/ex",
                "act_operators.rs",
            ),
            "integrals",
            &[KERNEL[1]],
            concat!(
                "Every Tet10/Tri6 matrix entry on all faces; independent simplex moments. No ",
                "Tet4 or public topology audit credit.",
            )
        ),
        witness!(
            "FEA-WITNESS-ACOUSTIC-INTERFACE",
            "runmat-analysis-fea",
            concat!(
                "physics::acoustic::canonical::tests::interface::two_fluid_interface_pressure",
                "_mean_velocity_and_power_refine",
            ),
            "crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/interface.rs",
            "regions",
            KERNEL,
            concat!(
                "Aligned two-medium slab, phase rotation and spatial refinement; not ",
                "arbitrary heterogeneous or curved-interface qualification.",
            )
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
    witnesses: &[
        witness!(
            "FEA-WITNESS-ACOUSTIC-LIFTING",
            "runmat-analysis-fea",
            concat!(
                "physics::acoustic::canonical::tests::radiation::exact_operators::complex_pre",
                "scribed_pressure_lifting_matches_independent_reduced_block_system",
            ),
            concat!(
                "crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/radiation/ex",
                "act_operators.rs",
            ),
            "lifting",
            &[KERNEL[1]],
            concat!(
                "Tet10 lifted complex system checked against independent dense oracle; no ",
                "generic solve or conditioning claim.",
            )
        ),
        witness!(
            "FEA-WITNESS-ACOUSTIC-BLOCH",
            "runmat-analysis-fea",
            concat!(
                "physics::acoustic::canonical::tests::periodic::anchored_bloch_plane_wave_pre",
                "ssure_mean_velocity_and_signed_intensity_refine",
            ),
            "crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/periodic.rs",
            "phase",
            KERNEL,
            concat!(
                "Anchored slab plane wave and declared paired planes; no unanchored ",
                "pole/global conditioning credit.",
            )
        ),
        witness!(
            "FEA-WITNESS-ACOUSTIC-RADIATION",
            "runmat-analysis-fea",
            concat!(
                "physics::acoustic::canonical::tests::radiation::local_radiation_pressure_pha",
                "se_mean_velocity_and_signed_power_refine",
            ),
            "crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/radiation.rs",
            "radiation",
            KERNEL,
            concat!(
                "Local radiation reference and original spatial gates; witness linkage is ",
                "not a passing verdict or nonlocal exterior-wave qualification.",
            )
        ),
        witness!(
            "FEA-WITNESS-ACOUSTIC-BOUNDARY-REJECT",
            "runmat-analysis-fea",
            concat!(
                "physics::acoustic::canonical::tests::periodic::periodic_rejects_unmatched_na",
                "tural_overlap_and_conflicting_pressure",
            ),
            "crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/periodic.rs",
            "reject",
            KERNEL,
            "Periodic pairing, overlap and conflicting pressure rejection only."
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
    witnesses: &[
        witness!(
            "FEA-WITNESS-ACOUSTIC-VOLUME",
            "runmat-analysis-fea",
            concat!(
                "physics::acoustic::canonical::tests::volume_velocity::total_volume_flow_area",
                "_phase_rotation_and_power_refine_against_continuum",
            ),
            "crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/volume_velocity.rs",
            "resultant",
            KERNEL,
            "Area scaling, complex phase and independent flow/power refinement on slab fixtures."
        ),
        witness!(
            "FEA-WITNESS-ACOUSTIC-INCIDENT",
            "runmat-analysis-fea",
            concat!(
                "physics::acoustic::canonical::tests::incident::incoming_pressure_reflection_",
                "transmission_and_net_power_refine",
            ),
            "crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/incident.rs",
            "incident",
            KERNEL,
            "Planar incident/impedance fixture, phase and net-power refinement only."
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
        "crates/runmat-analysis/fea/src/solve/uniqueness/mod.rs",
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
                "Source-independent original reduced-operator evidence establishes requested ",
                "uniqueness before accepting physical residual/power closure. An ",
                "unsuccessful sufficient check is indeterminate, not a singularity ",
                "diagnosis; no artificial damping or sampled small matrix substitutes for ",
                "the operator.",
            ),
            EvidenceKind::IndependentReference,
            KERNEL,
        ),
        obligation(
            "cold_claim",
            concat!(
                "Retained uniqueness records bind original controls, operator/result and ",
                "numerical receipt. Cold hashes or decoded producer claims never become ",
                "trusted mathematical proof; independent replay requires original operator ",
                "and admitted witness/candidate reconstruction.",
            ),
            EvidenceKind::OriginalStateComparison,
            KERNEL,
        ),
    ],
    witnesses: &[
        witness!(
            "FEA-WITNESS-ACOUSTIC-DUCT",
            "runmat-analysis-fea",
            concat!(
                "physics::acoustic::canonical::tests::continuum::tet10_traveling_wave_phase_p",
                "ressure_velocity_and_power_converge",
            ),
            "crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/continuum.rs",
            "duct",
            &[KERNEL[1]],
            concat!(
                "Tet10 slab at two frequencies, three refinements; exact traveling-wave ",
                "references, not all acoustic geometries.",
            )
        ),
        witness!(
            "FEA-WITNESS-ACOUSTIC-CAVITY",
            "runmat-analysis-fea",
            concat!(
                "physics::acoustic::canonical::tests::continuum::tet4_and_tet10_forced_rigid_",
                "cavity_converge_on_both_sides_of_axial_resonance",
            ),
            "crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/continuum.rs",
            "cavity",
            KERNEL,
            concat!(
                "Forced axial cavity at 0.9/1.1 times first pole, phase/pressure/energy ",
                "convergence; no complete multidimensional modal-spectrum credit.",
            )
        ),
        witness!(
            "FEA-WITNESS-ACOUSTIC-POLE",
            "runmat-analysis-fea",
            concat!(
                "physics::acoustic::canonical::response::reduced::tests::uniqueness::original",
                "_canonical_lossless_discrete_pole_is_not_proved_by_zero_rhs",
            ),
            concat!(
                "crates/runmat-analysis/fea/src/physics/acoustic/canonical/response/reduced/t",
                "ests/uniqueness.rs",
            ),
            "uniqueness",
            &[KERNEL[0]],
            concat!(
                "Independent Tet4 antisymmetric discrete pole; requested diagnostic refuses ",
                "zero-RHS residual acceptance, without asserting exact floating rank.",
            )
        ),
        witness!(
            "FEA-WITNESS-ACOUSTIC-COLD-CLAIM",
            "runmat-analysis-fea",
            concat!(
                "execution::acoustic::tests::uniqueness::required_diagnostic_survives_solve_f",
                "ree_cold_lineage_and_tampering_fails",
            ),
            "crates/runmat-analysis/fea/src/execution/acoustic/tests/uniqueness.rs",
            "cold_claim",
            &[KERNEL[0]],
            concat!(
                "Retained producer claim/receipt provenance and tamper rejection; no cold ",
                "mathematical uniqueness reproof.",
            )
        ),
    ],
};
const ACCURACY: Requirement = Requirement {
    id: "FEA-ACOUSTIC-ACCURACY-001",
    title: "Mandatory acoustic accuracy gates production publication",
    stages: &[Stage::Acoustics, Stage::ExecutionContracts],
    acceptance: concat!(
        "Every production acoustic entry retains independently authenticated ",
        "original numerical evidence and its authored accuracy intent. Evaluate ",
        "mandatory field/convergence, physical balance, source realization and ",
        "requested resonance/uniqueness requirements without weakening tolerances or ",
        "inferring scientific adequacy from solver success. Missing, failed or ",
        "indeterminate mandatory accuracy blocks dependent production state and ",
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
                "Public production completion and dependent publication reject missing, ",
                "failed or indeterminate mandatory acoustic accuracy; release qualification ",
                "is not inferred from an experimental accepted result.",
            ),
            EvidenceKind::AccuracyAcceptance,
            PUBLIC,
        ),
    ],
    witnesses: &[witness!(
        "FEA-WITNESS-ACOUSTIC-ACCURACY-LINEAGE",
        "runmat-analysis-execution",
        concat!(
            "tests::acoustic_accuracy::acoustic_accuracy_admission_binds_retained_result_",
            "and_original_intent",
        ),
        "crates/runmat-analysis/execution/src/tests/acoustic_accuracy.rs",
        "lineage",
        BRIDGE,
        concat!(
            "Both-order accepted-output bridge fixture validates exact context and ",
            "rejects production relabeling; not actual native worker or full public gate ",
            "evidence.",
        )
    )],
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
                "blocking release criteria pass before production advertisement.",
            ),
            EvidenceKind::AccuracyAcceptance,
            EXECUTION,
        ),
    ],
    witnesses: &[
        witness!(
            "FEA-WITNESS-ACOUSTIC-FANIN",
            "runmat-execution-runner-native",
            concat!(
                "analysis_kernel::acoustic_frequency_assembly::tests::native_admitted_frequen",
                "cy_assembly_reopens_original_complete_fields_and_metadata",
            ),
            concat!(
                "crates/runmat-execution-runner-native/src/analysis_kernel/acoustic_frequency",
                "_assembly/tests.rs",
            ),
            "inventory",
            NATIVE,
            concat!(
                "Tet4/Tet10, two/three serial-native frequencies, reversed completion and ",
                "cold filesystem fields; no actual parallel/distributed/GPU or ",
                "producer-termination claim.",
            )
        ),
        witness!(
            "FEA-WITNESS-ACOUSTIC-BUDGET",
            "runmat-execution-runner-native",
            concat!(
                "analysis_kernel::acoustic_harmonic::tests::native_acoustic_rejects_valid_but",
                "_substituted_resource_plan",
            ),
            "crates/runmat-execution-runner-native/src/analysis_kernel/acoustic_harmonic/tests.rs",
            "admission",
            &[NATIVE[0]],
            concat!(
                "Tet4 native worker plan substitution rejection; not measured RSS, ",
                "whole-family lifetime or complete cancellation evidence.",
            )
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
