//! Standing requirements for complete accepted nonlinear state and cold continuation.
//! State preservation does not close constitutive accuracy, adaptation or publication.
use crate::requirement::*;
const J2: &[Envelope] = &[
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet4,
        ExecutionEnvelope::PreparedController,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet10,
        ExecutionEnvelope::PreparedController,
    ),
];
const CONTACT: &[Envelope] = &[
    envelope(
        Formulation::SmallStrainJ2PenaltyCoulombContact,
        MeshOrder::Tet4,
        ExecutionEnvelope::PreparedController,
    ),
    envelope(
        Formulation::SmallStrainJ2PenaltyCoulombContact,
        MeshOrder::Tet10,
        ExecutionEnvelope::PreparedController,
    ),
];
const J2_NATIVE: &[Envelope] = &[
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet4,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet10,
        ExecutionEnvelope::NativeProcess,
    ),
];
const PREPARED: &[Envelope] = &[J2[0], J2[1], CONTACT[0], CONTACT[1]];
const NATIVE: &[Envelope] = &[
    J2_NATIVE[0],
    J2_NATIVE[1],
    envelope(
        Formulation::SmallStrainJ2PenaltyCoulombContact,
        MeshOrder::Tet4,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::SmallStrainJ2PenaltyCoulombContact,
        MeshOrder::Tet10,
        ExecutionEnvelope::NativeProcess,
    ),
];
pub(crate) const ACCEPTED_HISTORY: Requirement = Requirement {
    id: "FEA-NONLINEAR-STATE-001",
    title: "Complete original accepted nonlinear history through cold continuation",
    stages: &[Stage::NonlinearContact],
    acceptance: concat!(
        "Preserve complete original material/contact accepted history in immutable ",
        "canonical checkpoint artifacts. Restore cold initial outputs, validate ",
        "receiving-state physical compatibility, retain controller chronology and ",
        "survive native producer termination within each declared envelope. Element ",
        "averages and digest-only checks cannot replace original point-state ",
        "authority. Checkpoints and incomplete fields must not be presented as ",
        "completed converged base states.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/solve/nonlinear/solid.rs",
        "crates/runmat-analysis/fea/src/solve/nonlinear/driver/continuation.rs",
        "crates/runmat-execution-runner-native/src/analysis_kernel/nonlinear_block.rs",
    ],
    dependencies: &[
        concat!(
            "Independent J2/contact constitutive, equilibrium and mesh/load/penalty ",
            "accuracy must satisfy their mandatory acceptance criteria.",
        ),
        concat!(
            "Whole task resource admission, supported public contact controls and ",
            "mandatory accuracy-gated semantic publication.",
        ),
        concat!(
            "Supported feature/initial transitions and separately qualified adaptation ",
            "transfer; no transfer coverage inferred here.",
        ),
        concat!(
            "Typed state-transition manifests retain producer-step, topology/partition, ",
            "field/history, reference, constraint, precision and completeness lineage; ",
            "dependent steps are admitted only after immutable compatible state ",
            "publication. Activation, retry, restart and branching retain distinct ",
            "meanings and identities.",
        ),
    ],
    obligations: &[
        obligation(
            "points",
            concat!(
                "Persist every original ordered integration-point constitutive state and ",
                "complete contact active/friction history in immutable canonical artifacts, ",
                "with accepted displacement, load parameter and cumulative physical energy ",
                "state. Equilibrium and recovery consume that same accepted point history; ",
                "element averages cannot replace restart authority.",
            ),
            EvidenceKind::OriginalStateComparison,
            PREPARED,
        ),
        obligation(
            "rollback",
            concat!(
                "Rejected Newton/line-search/cutback trials and cancellation must not mutate ",
                "accepted material/contact history or publish an accepted suffix.",
            ),
            EvidenceKind::RejectionInvariant,
            PREPARED,
        ),
        obligation(
            "initial_outputs",
            concat!(
                "A cold block restores initial plastic/contact outputs from its original ",
                "accepted point history rather than zeroing or reinitializing it.",
            ),
            EvidenceKind::ColdReplay,
            PREPARED,
        ),
        obligation(
            "cold_validation",
            concat!(
                "Reject corrupted, reordered, missing, incompatible or rehashed substituted ",
                "scope/history before accepting a cold continuation. Validate the receiving ",
                "formulation, topology, integration inventory, properties, constraints, ",
                "reference conventions, point ordering, accepted load parameter and physical ",
                "state; digests alone are insufficient.",
            ),
            EvidenceKind::RejectionInvariant,
            PREPARED,
        ),
        obligation(
            "chronology",
            concat!(
                "Retain original scheduled target and cutbacks, cumulative failed/replay ",
                "work and caps, accepted/output ordinals and predecessor order across cold ",
                "blocks.",
            ),
            EvidenceKind::ControllerChronology,
            PREPARED,
        ),
        obligation(
            "producer_exit",
            concat!(
                "A new native consumer after producer termination resumes original complete ",
                "history and outputs for homogeneous and nonuniform supported bodies, ",
                "without the producer memory.",
            ),
            EvidenceKind::ProcessLifecycle,
            NATIVE,
        ),
    ],
};
const NONLINEAR_DOMAIN: &[Envelope] = &[
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const J2_OPERATOR_DOMAIN: &[Envelope] = &[
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const CONTACT_OPERATOR_DOMAIN: &[Envelope] = &[
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const FEA_NONLINEAR_OPERATOR_001: Requirement = Requirement {
    id: "FEA-NONLINEAR-OPERATOR-001",
    title: "State-dependent nonlinear mechanics and supported plastic laws",
    stages: &[Stage::NonlinearContact],
    acceptance: concat!(
        "State-dependent nonlinear mechanics and supported plastic laws must satisfy ",
        "all applicable supported-envelope criteria. Independent physical, semantic, ",
        "accuracy, resource and publication evidence are separate; missing, failed ",
        "or indeterminate mandatory evidence prevents acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/solve/nonlinear/solid.rs"],
    obligations: &[
        obligation(
            "kinematics_tangent_newton",
            concat!(
                "Define nonlinear kinematics supported initially and reject unsupported ",
                "combinations. Assemble internal force and consistent or documented ",
                "algorithmic tangent from current state. Implement Newton iteration with ",
                "real residual and increment criteria, line search, and load/time stepping.",
            ),
            EvidenceKind::IndependentReference,
            J2_OPERATOR_DOMAIN,
        ),
        obligation(
            "point_state_constitutive_laws",
            concat!(
                "Store integration-point constitutive state. Implement return mapping and ",
                "tangent for each explicitly supported plastic law. Feed plastic state into ",
                "equilibrium and recovery.",
            ),
            EvidenceKind::IndependentReference,
            J2_OPERATOR_DOMAIN,
        ),
        obligation(
            "canonical_state_physical_cutbacks",
            concat!(
                "Persist state in canonical artifacts suitable for restart and adaptation ",
                "transfer. Add cutback/retry policies driven by physical convergence rather ",
                "than synthetic scalar damping.",
            ),
            EvidenceKind::IndependentReference,
            J2_OPERATOR_DOMAIN,
        ),
        obligation(
            "references",
            concat!(
                "Evidence includes constitutive point tests, elastoplastic bars, cyclic ",
                "paths where supported, large-displacement beams, limit-point cases only ",
                "when the continuation method supports them, and mesh/load-step convergence.",
            ),
            EvidenceKind::IndependentReference,
            J2_OPERATOR_DOMAIN,
        ),
    ],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
const FEA_CONTACT_OPERATOR_001: Requirement = Requirement {
    id: "FEA-CONTACT-OPERATOR-001",
    title: "Authored surface contact, search, friction and coupled inequality equilibrium",
    stages: &[Stage::NonlinearContact],
    acceptance: concat!(
        "Authored surface contact, search, friction and coupled inequality ",
        "equilibrium must satisfy all applicable supported-envelope criteria. ",
        "Independent physical, semantic, accuracy, resource and publication evidence ",
        "are separate; missing, failed or indeterminate mandatory evidence prevents ",
        "acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/physics/structural/contact/mod.rs"],
    obligations: &[
        obligation(
            "surface_search_gap",
            concat!(
                "Resolve authored primary and secondary surface identities through the ",
                "shared topology projection. Build deterministic broad-phase and ",
                "narrow-phase search. Compute oriented normals, closest projection, gap, ",
                "quadrature, and active-set state.",
            ),
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN,
        ),
        obligation(
            "normal_friction_equilibrium",
            concat!(
                "Implement a documented normal enforcement method and friction law. Assemble ",
                "contact residual and tangent into nonlinear equilibrium. Update contact ",
                "state as geometry changes.",
            ),
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN,
        ),
        obligation(
            "history_search_diagnostics",
            concat!(
                "Preserve contact history across restart and supported adaptation. Diagnose ",
                "initial penetration, unmatched surfaces, normal ambiguity, search ",
                "exhaustion, and parameter sensitivity. The contact system owns ",
                "deterministic surface pairing and search updates.",
            ),
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN,
        ),
        obligation(
            "gap_active_enforcement",
            concat!(
                "The contact system owns gap and normal definitions, active-set or ",
                "complementarity state, and normal enforcement.",
            ),
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN,
        ),
        obligation(
            "friction_tangent_history",
            concat!(
                "The contact system owns tangential/friction state, residual and tangent ",
                "contributions, and history/restart data.",
            ),
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN,
        ),
        obligation(
            "complementarity_penetration",
            "The contact system owns complementarity and penetration evidence.",
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN,
        ),
        obligation(
            "references",
            concat!(
                "Evidence includes gap closure, patch contact, Hertz-type normal contact, ",
                "sliding/friction, objectivity, action-reaction balance, and ",
                "mesh/contact-parameter convergence.",
            ),
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN,
        ),
        obligation(
            "inequality",
            concat!(
                "Authored contact interfaces compile to nonlinear inequalities, never the ",
                "equality matrix. Diagnose tied/contacting surface overlap without erasing ",
                "their mathematical distinction. Stable authored identities survive changing ",
                "geometry while normals, pairings and active state update in the owning ",
                "nonlinear formulation.",
            ),
            EvidenceKind::RejectionInvariant,
            CONTACT_OPERATOR_DOMAIN,
        ),
    ],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
const FEA_NONLINEAR_GOVERNANCE_001: Requirement = Requirement {
    id: "FEA-NONLINEAR-GOVERNANCE-001",
    title: "Converged nonlinear states, supported laws and independent qualification",
    stages: &[Stage::NonlinearContact],
    acceptance: concat!(
        "Converged nonlinear states, supported laws and independent qualification ",
        "must satisfy all applicable supported-envelope criteria. Independent ",
        "physical, semantic, accuracy, resource and publication evidence are ",
        "separate; missing, failed or indeterminate mandatory evidence prevents ",
        "acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/pipeline/nonlinear.rs"],
    obligations: &[
        obligation(
            "operator_constitutive_contact",
            concat!(
                "Implement nonlinear state, residual, and tangent assembly. Implement ",
                "supported plastic integration and history. Implement contact search, ",
                "enforcement, friction, and tangent coupling.",
            ),
            EvidenceKind::AccuracyAcceptance,
            NONLINEAR_DOMAIN,
        ),
        obligation(
            "inequality_restart_converged_base",
            concat!(
                "Validate contact as a nonlinear inequality system alongside, but distinct ",
                "from, equality constraints. Implement restart and supported transfer of ",
                "nonlinear state. Publish only converged nonlinear base states that satisfy ",
                "the declared transition envelope; distinguish physical state from an ",
                "initial guess or incomplete checkpoint.",
            ),
            EvidenceKind::AccuracyAcceptance,
            NONLINEAR_DOMAIN,
        ),
        obligation(
            "downstream_checkpoints_release",
            concat!(
                "Qualify selected nonlinear base states for buckling or prestressed ",
                "downstream procedures only after their tangent, contact, material-history, ",
                "and transition evidence passes. Add nonlinear task checkpoints and bounded ",
                "retries. Add release evidence and documentation for each formulation ",
                "separately.",
            ),
            EvidenceKind::AccuracyAcceptance,
            NONLINEAR_DOMAIN,
        ),
        obligation(
            "executable_laws_independent_physics",
            concat!(
                "Keep state-dependent follower loads, controllers, and constitutive/user ",
                "laws outside ordinary excitation callbacks; reject them until the separate ",
                "executable-law ABI and tangent contract exist. Equilibrium is satisfied by ",
                "the implemented state-dependent operator. Constitutive and contact ",
                "benchmarks pass independently.",
            ),
            EvidenceKind::AccuracyAcceptance,
            NONLINEAR_DOMAIN,
        ),
        obligation(
            "sensitivity_conflict_diagnostics",
            concat!(
                "Contact complementarity, equality-constraint violation, and ",
                "overlap/conflict diagnostics pass independently. Load-step, mesh, and ",
                "contact-parameter sensitivity are governed.",
            ),
            EvidenceKind::AccuracyAcceptance,
            NONLINEAR_DOMAIN,
        ),
    ],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    ACCEPTED_HISTORY,
    FEA_NONLINEAR_OPERATOR_001,
    FEA_CONTACT_OPERATOR_001,
    FEA_NONLINEAR_GOVERNANCE_001,
];
