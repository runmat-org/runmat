//! Standing acceptance for modal, prestressed, stability and forced structural dynamics.
use crate::requirement::*;
const DOMAIN: &[Envelope] = &[
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::MeshIndependent,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::Beam,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::Shell,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::Beam,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::Shell,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::LinearStructuralDynamics,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const MODAL_DOMAIN: &[Envelope] = &[
    envelope(
        Formulation::ModalStructural,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::ModalStructural,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::ModalStructural,
        MeshOrder::Beam,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::ModalStructural,
        MeshOrder::Shell,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::ModalStructural,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::ModalStructural,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::ModalStructural,
        MeshOrder::Beam,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::ModalStructural,
        MeshOrder::Shell,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::ModalStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::ModalStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::ModalStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::ModalStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        Formulation::ModalStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::ModalStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const PRESTRESS_DOMAIN: &[Envelope] = &[
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::Beam,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::Shell,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::Beam,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::Shell,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::PrestressedStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::Beam,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::Shell,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::Beam,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::Shell,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::StructuralStability,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const FEA_DYNAMICS_MODAL_001: Requirement = Requirement {
    id: "FEA-DYNAMICS-MODAL-001",
    title: "Physical modal extraction, eigenspaces and reusable participation state",
    stages: &[Stage::Dynamics],
    acceptance: concat!(
        "Physical modal extraction, eigenspaces and reusable participation state ",
        "must satisfy all applicable supported-envelope criteria. Independent ",
        "physical, semantic, accuracy, resource and publication evidence are ",
        "separate; missing, failed or indeterminate mandatory evidence prevents ",
        "acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/pipeline/modal.rs"],
    obligations: &[
        obligation(
            "physical_mass_base_state",
            concat!(
                "Use the physical structural stiffness and mass operators. Treat unstressed ",
                "modal extraction and prestressed modal extraction as distinct typed ",
                "procedures; the latter consumes a validated converged base state through ",
                "the sequence contract. Support documented consistent/lumped mass selection.",
            ),
            EvidenceKind::IndependentReference,
            MODAL_DOMAIN,
        ),
        obligation(
            "eigenpairs_eigenspaces",
            concat!(
                "Handle constraints, rigid modes, repeated and clustered eigenvalues. ",
                "Mass-normalize modes and report residual and orthogonality measures. Make ",
                "mode ordering deterministic without pretending that an arbitrary basis ",
                "inside a repeated eigenspace is unique.",
            ),
            EvidenceKind::IndependentReference,
            MODAL_DOMAIN,
        ),
        obligation(
            "participation_partitioning",
            concat!(
                "Publish participation factors, effective modal mass by direction, and ",
                "cumulative captured mass for later forced-response procedures. Partition ",
                "matrix-vector work only after serial eigensolver evidence is complete.",
            ),
            EvidenceKind::IndependentReference,
            MODAL_DOMAIN,
        ),
        obligation(
            "references",
            concat!(
                "Evidence includes axial bar and beam modes, three-dimensional solids, ",
                "repeated modes, mass orthogonality, residuals, and mesh convergence.",
            ),
            EvidenceKind::IndependentReference,
            MODAL_DOMAIN,
        ),
    ],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
const FEA_DYNAMICS_PRESTRESS_001: Requirement = Requirement {
    id: "FEA-DYNAMICS-PRESTRESS-001",
    title: "Validated prestressed base states and stability perturbation",
    stages: &[Stage::Dynamics],
    acceptance: concat!(
        "Validated prestressed base states and stability perturbation must satisfy ",
        "all applicable supported-envelope criteria. Independent physical, semantic, ",
        "accuracy, resource and publication evidence are separate; missing, failed ",
        "or indeterminate mandatory evidence prevents acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/tests/prestressed_frequency.rs"],
    obligations: &[
        obligation(
            "complete_base_operator",
            concat!(
                "Consume only a complete, converged structural base state from an explicit ",
                "sequence transition. Support a documented initial envelope of linear static ",
                "preload and selected converged nonlinear preload states for which a ",
                "consistent tangent and geometric stiffness are available. Assemble and ",
                "retain material, tangent, and geometric-stiffness contributions with ",
                "separate identities and sign conventions.",
            ),
            EvidenceKind::IndependentReference,
            PRESTRESS_DOMAIN,
        ),
        obligation(
            "buckling_modal_lineage",
            concat!(
                "Implement eigenvalue buckling as a linear perturbation about the base ",
                "state, with typed load-pattern scaling and explicit treatment of fixed, ",
                "proportional, follower, and unsupported loads. Report signed load factors, ",
                "normalized buckling modes, eigen-residuals, clustered/repeated eigenspaces, ",
                "base-load attribution, and the assumptions that prevent interpreting the ",
                "result as nonlinear post-buckling capacity. Implement prestressed modal ",
                "extraction from the supported tangent and mass operators, preserving the ",
                "base-state constraint and contact policy.",
            ),
            EvidenceKind::IndependentReference,
            PRESTRESS_DOMAIN,
        ),
        obligation(
            "unsupported_linearization",
            concat!(
                "Permit prestressed direct or modal harmonic response only after both the ",
                "base-state transition and the corresponding harmonic formulation are ",
                "independently verified. Reject unsupported material history, changing ",
                "contact status, follower-load linearization, topology change, loss of ",
                "tangent symmetry, or unstable base states rather than silently reverting to ",
                "an unstressed operator. Keep geometric imperfection analysis, nonlinear ",
                "buckling, collapse, and post-buckling outside the supported envelope until ",
                "separately designed and validated.",
            ),
            EvidenceKind::IndependentReference,
            PRESTRESS_DOMAIN,
        ),
        obligation(
            "references",
            concat!(
                "Evidence includes column and frame buckling references, mesh convergence, ",
                "sign and load-scaling cases, prestress-induced modal frequency shifts, ",
                "tension stiffening, compression softening, repeated modes, ",
                "tangent/geometric-stiffness checks, and sequence/restart equivalence.",
            ),
            EvidenceKind::IndependentReference,
            PRESTRESS_DOMAIN,
        ),
    ],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
const FEA_DYNAMICS_RESPONSE_001: Requirement = Requirement {
    id: "FEA-DYNAMICS-RESPONSE-001",
    title: "Distinct second-order direct and modal dynamic procedures",
    stages: &[Stage::Dynamics],
    acceptance: concat!(
        "Distinct second-order direct and modal dynamic procedures must satisfy all ",
        "applicable supported-envelope criteria. Independent physical, semantic, ",
        "accuracy, resource and publication evidence are separate; missing, failed ",
        "or indeterminate mandatory evidence prevents acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/solve/dynamics/mod.rs"],
    obligations: &[
        obligation(
            "second_order_equilibrium_excitation",
            concat!(
                "Replace first-order relaxation with a documented second-order integrator, ",
                "initially generalized-alpha or Newmark with explicit parameter policy. ",
                "Solve M a + C v + K u = f(t) with physical mass, damping, initial ",
                "displacement, initial velocity, and load history. Support force, moment, ",
                "traction, pressure, body/base displacement, velocity, and acceleration ",
                "histories through the shared typed excitation system.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "direct_modal_procedures",
            concat!(
                "Implement modal transient response, modal harmonic response, and modal ",
                "response-spectrum analysis as procedures distinct from eigenmode ",
                "extraction. Implement direct harmonic response from the documented complex ",
                "dynamic-stiffness operator, physical damping model, frequency-dependent ",
                "excitation, and reusable pattern/value identities; do not route it through ",
                "modal reduction implicitly. Support direct transient response independently ",
                "of modal transient response and compare both inside their shared validity ",
                "envelope.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "spectrum_random_recovery",
            concat!(
                "Implement typed response spectra, modal and directional combination, ",
                "participation/mass capture, residual-mode and missing-mass policy, and ",
                "attributable combination evidence. Support random vibration only from ",
                "governed PSD/cross-PSD inputs with documented correlation and output ",
                "statistics. Recover velocity and acceleration from the integration state, ",
                "not post-hoc finite differences alone.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "time_policy_energy_checkpoints",
            concat!(
                "Support adaptive time stepping only through documented local error or ",
                "convergence criteria. Report energy behavior, numerical dissipation, ",
                "stability parameters, rejected steps, and solver convergence. Use ",
                "checkpointed temporal blocks for long durable/remote runs.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "references",
            concat!(
                "Evidence includes undamped and damped oscillators, impulse and base-motion ",
                "response, callable/table equivalence, modal superposition comparisons, SRS ",
                "construction references, SRSS/CQC and directional-combination references, ",
                "missing-mass cases, random-response references where supported, energy ",
                "behavior, and time-step convergence.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
    ],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
const FEA_DYNAMICS_GOVERNANCE_001: Requirement = Requirement {
    id: "FEA-DYNAMICS-GOVERNANCE-001",
    title: "Governed dynamic procedures, checkpoints and distributed evidence",
    stages: &[Stage::Dynamics],
    acceptance: concat!(
        "Governed dynamic procedures, checkpoints and distributed evidence must ",
        "satisfy all applicable supported-envelope criteria. Independent physical, ",
        "semantic, accuracy, resource and publication evidence are separate; ",
        "missing, failed or indeterminate mandatory evidence prevents ",
        "acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/pipeline/modal.rs"],
    obligations: &[
        obligation(
            "modal_preload_buckling",
            concat!(
                "Complete modal analysis on physical mass/stiffness. Implement typed ",
                "linear-static structural base-state transitions, ",
                "tangent/geometric-stiffness construction, and linearization evidence; leave ",
                "the validated extension point for selected nonlinear producer states. ",
                "Implement eigenvalue buckling for the supported preload and load-scaling ",
                "envelope; explicitly reject nonlinear post-buckling implications.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "prestress_direct_dynamics",
            concat!(
                "Implement prestressed modal extraction and verify stress ",
                "stiffening/softening against unstressed references. Implement second-order ",
                "direct structural transient dynamics. Implement direct harmonic response ",
                "with complex dynamic stiffness, documented damping, and frequency-dependent ",
                "excitation.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "participation_modal_kinematics",
            concat!(
                "Publish modal participation factors, effective modal mass by direction, ",
                "cumulative captured mass, truncation evidence, and reusable modal-basis ",
                "artifacts. Implement shared time-history force and base-motion excitation ",
                "with homogeneous modal constraint rules and time-dependent affine ",
                "constraint kinematics. Implement modal transient and modal harmonic ",
                "response as procedures distinct from modal extraction and their direct ",
                "counterparts.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "spectra_random_input",
            concat!(
                "Implement SRS construction from a base-acceleration history as a general ",
                "signal operation and typed response-spectrum admission as analysis input. ",
                "Implement modal response-spectrum analysis with governed SRSS/CQC, ",
                "directional combination, residual-mode, rigid-response, and missing-mass ",
                "policies. Implement random vibration from typed PSD/cross-PSD inputs with ",
                "correlation, modal truncation, and output-statistics contracts.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "checkpoint_procedure_references",
            concat!(
                "Add restart/checkpoint, transient continuation, immutable base-state ",
                "branching, and temporal-block execution. Add governed buckling, ",
                "prestressed, modal, direct/modal time-domain, direct/modal harmonic, SRS, ",
                "response-spectrum, and random-response evidence. Add distributed ",
                "frequency/load-case/spectrum-direction/sequence-branch parallelism; add ",
                "operator partitioning only if measurements justify it.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "distributed_modal_gates",
            concat!(
                "Define cooperative worker-group, partition epoch, halo, collective, and ",
                "group-restart contracts before the first distributed operator ",
                "implementation. Modal residual, orthogonality, and convergence gates pass. ",
                "Buckling residual, load-scaling, sign, mesh-convergence, base-state, and ",
                "tangent/geometric-stiffness gates pass within the supported envelope.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "prestress_combination_equivalence",
            concat!(
                "Prestressed modal and harmonic results retain complete base-state lineage ",
                "and pass independent frequency/response references. Participation, ",
                "effective-mass, truncation, modal/directional combination, and missing-mass ",
                "evidence passes for each promoted modal-response procedure. Direct and ",
                "modal transient/harmonic procedures remain distinct, and their shared ",
                "validity cases agree within governed tolerances.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "dynamic_balance_generated_inputs",
            concat!(
                "Dynamic equilibrium, energy, time-step, frequency-grid, and complex ",
                "power/phase gates pass. Explicit tables, RunMat closures, and named .fea ",
                "project generators produce equivalent governed histories and responses. ",
                "Prescribed motion produces consistent displacement, velocity, acceleration, ",
                "reactions, and work.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "motion_restart_branches",
            concat!(
                "Durable and remote restart preserve logical results. Uninterrupted and ",
                "restarted continuations agree, and independent branches cannot mutate or ",
                "relabel the producer state. Cooperative distributed evidence distinguishes ",
                "task distribution from a tightly coupled solve and fences stale group ",
                "epochs.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
    ],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_DYNAMICS_MODAL_001,
    FEA_DYNAMICS_PRESTRESS_001,
    FEA_DYNAMICS_RESPONSE_001,
    FEA_DYNAMICS_GOVERNANCE_001,
];
