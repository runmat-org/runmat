use std::sync::Once;

use runmat_analysis_core::{
    InitialStateConflictPolicy, ReferenceStatePolicy, SequenceDependencyReason, StateCompleteness,
    StateKind, StatePersistence, StateTransitionMethod, StateUse, TopologyTransitionPolicy,
};

use super::super::*;
use crate::builtins::fea::study_values::{
    from_typed_value, register_immutable_class, to_typed_value, VALUE_FIELD,
};

pub(super) const STATE_KIND_CLASS: &str = "fea.StateKind";
pub(super) const STATE_COMPLETENESS_CLASS: &str = "fea.StateCompleteness";
pub(super) const STATE_PERSISTENCE_CLASS: &str = "fea.StatePersistence";
pub(super) const STATE_USE_CLASS: &str = "fea.StateUse";
pub(super) const INITIAL_CONFLICT_CLASS: &str = "fea.InitialStateConflictPolicy";
pub(super) const DEPENDENCY_REASON_CLASS: &str = "fea.SequenceDependencyReason";
pub(super) const TRANSITION_METHOD_CLASS: &str = "fea.StateTransitionMethod";
pub(super) const TOPOLOGY_TRANSITION_CLASS: &str = "fea.TopologyTransitionPolicy";
pub(super) const REFERENCE_STATE_CLASS: &str = "fea.ReferenceStatePolicy";

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const CHOICE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Closed FEA study-policy choices have no numeric inputs.",
};

macro_rules! choice {
    ($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $label:literal, $summary:literal, $class:expr, $value:expr) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &NO_INPUTS,
            outputs: &OUT_ANY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
        #[runtime_builtin(
                                    name = $builtin,
                                    category = "fea",
                                    summary = $summary,
                                    keywords = "fea,study,sequence,state,policy,typed",
                                    descriptor($descriptor),
                                    integer_audit(CHOICE_INTEGER_AUDIT),
                                    builtin_path = "crate::builtins::fea::study_policies::sequence"
                                )]
        pub async fn $function() -> BuiltinResult<Value> {
            to_typed_value($builtin, $class, $value)
        }
    };
}

choice!(
    state_structural_primary,
    STATE_STRUCTURAL_PRIMARY_DESCRIPTOR,
    STATE_STRUCTURAL_PRIMARY_SIGNATURES,
    "fea.StateKind.structuralPrimary",
    "kind = fea.StateKind.structuralPrimary()",
    "Select a complete structural primary state.",
    STATE_KIND_CLASS,
    StateKind::StructuralPrimary
);
choice!(
    state_structural_kinematics,
    STATE_STRUCTURAL_KINEMATICS_DESCRIPTOR,
    STATE_STRUCTURAL_KINEMATICS_SIGNATURES,
    "fea.StateKind.structuralKinematics",
    "kind = fea.StateKind.structuralKinematics()",
    "Select a structural kinematics state.",
    STATE_KIND_CLASS,
    StateKind::StructuralKinematics
);
choice!(
    state_structural_stress_strain,
    STATE_STRUCTURAL_STRESS_STRAIN_DESCRIPTOR,
    STATE_STRUCTURAL_STRESS_STRAIN_SIGNATURES,
    "fea.StateKind.structuralStressStrain",
    "kind = fea.StateKind.structuralStressStrain()",
    "Select a structural stress-strain state.",
    STATE_KIND_CLASS,
    StateKind::StructuralStressStrain
);
choice!(
    state_geometric_stiffness,
    STATE_GEOMETRIC_STIFFNESS_DESCRIPTOR,
    STATE_GEOMETRIC_STIFFNESS_SIGNATURES,
    "fea.StateKind.geometricStiffness",
    "kind = fea.StateKind.geometricStiffness()",
    "Select a geometric-stiffness state.",
    STATE_KIND_CLASS,
    StateKind::GeometricStiffness
);
choice!(
    state_material_history,
    STATE_MATERIAL_HISTORY_DESCRIPTOR,
    STATE_MATERIAL_HISTORY_SIGNATURES,
    "fea.StateKind.materialHistory",
    "kind = fea.StateKind.materialHistory()",
    "Select a material-history state.",
    STATE_KIND_CLASS,
    StateKind::MaterialHistory
);
choice!(
    state_contact_history,
    STATE_CONTACT_HISTORY_DESCRIPTOR,
    STATE_CONTACT_HISTORY_SIGNATURES,
    "fea.StateKind.contactHistory",
    "kind = fea.StateKind.contactHistory()",
    "Select a contact-history state.",
    STATE_KIND_CLASS,
    StateKind::ContactHistory
);
choice!(
    state_thermal,
    STATE_THERMAL_DESCRIPTOR,
    STATE_THERMAL_SIGNATURES,
    "fea.StateKind.thermal",
    "kind = fea.StateKind.thermal()",
    "Select a thermal state.",
    STATE_KIND_CLASS,
    StateKind::Thermal
);
choice!(
    state_electromagnetic_static,
    STATE_ELECTROMAGNETIC_STATIC_DESCRIPTOR,
    STATE_ELECTROMAGNETIC_STATIC_SIGNATURES,
    "fea.StateKind.electromagneticStatic",
    "kind = fea.StateKind.electromagneticStatic()",
    "Select a static electromagnetic state.",
    STATE_KIND_CLASS,
    StateKind::ElectromagneticStatic
);
choice!(
    state_electromagnetic_complex,
    STATE_ELECTROMAGNETIC_COMPLEX_DESCRIPTOR,
    STATE_ELECTROMAGNETIC_COMPLEX_SIGNATURES,
    "fea.StateKind.electromagneticComplex",
    "kind = fea.StateKind.electromagneticComplex()",
    "Select a complex electromagnetic state.",
    STATE_KIND_CLASS,
    StateKind::ElectromagneticComplex
);
choice!(
    state_acoustic_complex,
    STATE_ACOUSTIC_COMPLEX_DESCRIPTOR,
    STATE_ACOUSTIC_COMPLEX_SIGNATURES,
    "fea.StateKind.acousticComplex",
    "kind = fea.StateKind.acousticComplex()",
    "Select a complex acoustic state.",
    STATE_KIND_CLASS,
    StateKind::AcousticComplex
);
choice!(
    state_fluid,
    STATE_FLUID_DESCRIPTOR,
    STATE_FLUID_SIGNATURES,
    "fea.StateKind.fluid",
    "kind = fea.StateKind.fluid()",
    "Select a fluid state.",
    STATE_KIND_CLASS,
    StateKind::Fluid
);
choice!(
    state_moving_mesh,
    STATE_MOVING_MESH_DESCRIPTOR,
    STATE_MOVING_MESH_SIGNATURES,
    "fea.StateKind.movingMesh",
    "kind = fea.StateKind.movingMesh()",
    "Select a moving-mesh state.",
    STATE_KIND_CLASS,
    StateKind::MovingMesh
);
choice!(
    state_modal_basis,
    STATE_MODAL_BASIS_DESCRIPTOR,
    STATE_MODAL_BASIS_SIGNATURES,
    "fea.StateKind.modalBasis",
    "kind = fea.StateKind.modalBasis()",
    "Select a modal-basis state.",
    STATE_KIND_CLASS,
    StateKind::ModalBasis
);
choice!(
    state_coupled_interface,
    STATE_COUPLED_INTERFACE_DESCRIPTOR,
    STATE_COUPLED_INTERFACE_SIGNATURES,
    "fea.StateKind.coupledInterface",
    "kind = fea.StateKind.coupledInterface()",
    "Select a coupled-interface state.",
    STATE_KIND_CLASS,
    StateKind::CoupledInterface
);
choice!(
    state_integrator,
    STATE_INTEGRATOR_DESCRIPTOR,
    STATE_INTEGRATOR_SIGNATURES,
    "fea.StateKind.integrator",
    "kind = fea.StateKind.integrator()",
    "Select a time-integrator state.",
    STATE_KIND_CLASS,
    StateKind::Integrator
);

choice!(
    completeness_complete,
    COMPLETENESS_COMPLETE_DESCRIPTOR,
    COMPLETENESS_COMPLETE_SIGNATURES,
    "fea.StateCompleteness.complete",
    "value = fea.StateCompleteness.complete()",
    "Require a complete state.",
    STATE_COMPLETENESS_CLASS,
    StateCompleteness::Complete
);
choice!(
    completeness_checkpoint,
    COMPLETENESS_CHECKPOINT_DESCRIPTOR,
    COMPLETENESS_CHECKPOINT_SIGNATURES,
    "fea.StateCompleteness.checkpoint",
    "value = fea.StateCompleteness.checkpoint()",
    "Select checkpoint completeness.",
    STATE_COMPLETENESS_CLASS,
    StateCompleteness::Checkpoint
);
choice!(
    completeness_partial,
    COMPLETENESS_PARTIAL_DESCRIPTOR,
    COMPLETENESS_PARTIAL_SIGNATURES,
    "fea.StateCompleteness.partial",
    "value = fea.StateCompleteness.partial()",
    "Select a partial state.",
    STATE_COMPLETENESS_CLASS,
    StateCompleteness::Partial
);

choice!(
    persistence_canonical,
    PERSISTENCE_CANONICAL_DESCRIPTOR,
    PERSISTENCE_CANONICAL_SIGNATURES,
    "fea.StatePersistence.canonical",
    "value = fea.StatePersistence.canonical()",
    "Retain state canonically.",
    STATE_PERSISTENCE_CLASS,
    StatePersistence::Canonical
);
choice!(
    persistence_until_dependents,
    PERSISTENCE_UNTIL_DEPENDENTS_DESCRIPTOR,
    PERSISTENCE_UNTIL_DEPENDENTS_SIGNATURES,
    "fea.StatePersistence.untilDependentsComplete",
    "value = fea.StatePersistence.untilDependentsComplete()",
    "Retain state until dependents complete.",
    STATE_PERSISTENCE_CLASS,
    StatePersistence::UntilDependentsComplete
);
choice!(
    persistence_checkpoint_only,
    PERSISTENCE_CHECKPOINT_ONLY_DESCRIPTOR,
    PERSISTENCE_CHECKPOINT_ONLY_SIGNATURES,
    "fea.StatePersistence.checkpointOnly",
    "value = fea.StatePersistence.checkpointOnly()",
    "Retain state only as a checkpoint.",
    STATE_PERSISTENCE_CLASS,
    StatePersistence::CheckpointOnly
);

choice!(
    use_physical_initial,
    USE_PHYSICAL_INITIAL_DESCRIPTOR,
    USE_PHYSICAL_INITIAL_SIGNATURES,
    "fea.StateUse.physicalInitialState",
    "value = fea.StateUse.physicalInitialState()",
    "Consume state as the physical initial state.",
    STATE_USE_CLASS,
    StateUse::PhysicalInitialState
);
choice!(
    use_linearization,
    USE_LINEARIZATION_DESCRIPTOR,
    USE_LINEARIZATION_SIGNATURES,
    "fea.StateUse.linearizationPoint",
    "value = fea.StateUse.linearizationPoint()",
    "Consume state as a linearization point.",
    STATE_USE_CLASS,
    StateUse::LinearizationPoint
);
choice!(
    use_prestress,
    USE_PRESTRESS_DESCRIPTOR,
    USE_PRESTRESS_SIGNATURES,
    "fea.StateUse.prestress",
    "value = fea.StateUse.prestress()",
    "Consume state as prestress.",
    STATE_USE_CLASS,
    StateUse::Prestress
);
choice!(
    use_initial_guess,
    USE_INITIAL_GUESS_DESCRIPTOR,
    USE_INITIAL_GUESS_SIGNATURES,
    "fea.StateUse.initialGuess",
    "value = fea.StateUse.initialGuess()",
    "Consume state only as an initial guess.",
    STATE_USE_CLASS,
    StateUse::InitialGuess
);
choice!(
    use_modal_basis,
    USE_MODAL_BASIS_DESCRIPTOR,
    USE_MODAL_BASIS_SIGNATURES,
    "fea.StateUse.modalBasis",
    "value = fea.StateUse.modalBasis()",
    "Consume state as a modal basis.",
    STATE_USE_CLASS,
    StateUse::ModalBasis
);
choice!(
    use_coupled_interface,
    USE_COUPLED_INTERFACE_DESCRIPTOR,
    USE_COUPLED_INTERFACE_SIGNATURES,
    "fea.StateUse.coupledInterface",
    "value = fea.StateUse.coupledInterface()",
    "Consume state at a coupled interface.",
    STATE_USE_CLASS,
    StateUse::CoupledInterface
);
choice!(
    use_restart,
    USE_RESTART_DESCRIPTOR,
    USE_RESTART_SIGNATURES,
    "fea.StateUse.restart",
    "value = fea.StateUse.restart()",
    "Consume state for restart.",
    STATE_USE_CLASS,
    StateUse::Restart
);

choice!(
    conflict_reject,
    CONFLICT_REJECT_DESCRIPTOR,
    CONFLICT_REJECT_SIGNATURES,
    "fea.InitialStateConflictPolicy.reject",
    "value = fea.InitialStateConflictPolicy.reject()",
    "Reject an initial-state conflict.",
    INITIAL_CONFLICT_CLASS,
    InitialStateConflictPolicy::Reject
);
choice!(
    conflict_replace,
    CONFLICT_REPLACE_DESCRIPTOR,
    CONFLICT_REPLACE_SIGNATURES,
    "fea.InitialStateConflictPolicy.replaceInitialCondition",
    "value = fea.InitialStateConflictPolicy.replaceInitialCondition()",
    "Replace a conflicting initial condition.",
    INITIAL_CONFLICT_CLASS,
    InitialStateConflictPolicy::ReplaceInitialCondition
);
choice!(
    conflict_augment,
    CONFLICT_AUGMENT_DESCRIPTOR,
    CONFLICT_AUGMENT_SIGNATURES,
    "fea.InitialStateConflictPolicy.augmentDisjointQuantities",
    "value = fea.InitialStateConflictPolicy.augmentDisjointQuantities()",
    "Augment only disjoint initial-state quantities.",
    INITIAL_CONFLICT_CLASS,
    InitialStateConflictPolicy::AugmentDisjointQuantities
);

choice!(
    dependency_state_transfer,
    DEPENDENCY_STATE_TRANSFER_DESCRIPTOR,
    DEPENDENCY_STATE_TRANSFER_SIGNATURES,
    "fea.SequenceDependencyReason.stateTransfer",
    "value = fea.SequenceDependencyReason.stateTransfer()",
    "Order steps for state transfer.",
    DEPENDENCY_REASON_CLASS,
    SequenceDependencyReason::StateTransfer
);
choice!(
    dependency_authored,
    DEPENDENCY_AUTHORED_DESCRIPTOR,
    DEPENDENCY_AUTHORED_SIGNATURES,
    "fea.SequenceDependencyReason.authoredOrdering",
    "value = fea.SequenceDependencyReason.authoredOrdering()",
    "Apply explicit authored ordering.",
    DEPENDENCY_REASON_CLASS,
    SequenceDependencyReason::AuthoredOrdering
);
choice!(
    dependency_interface,
    DEPENDENCY_INTERFACE_DESCRIPTOR,
    DEPENDENCY_INTERFACE_SIGNATURES,
    "fea.SequenceDependencyReason.sharedInterface",
    "value = fea.SequenceDependencyReason.sharedInterface()",
    "Order steps sharing an interface.",
    DEPENDENCY_REASON_CLASS,
    SequenceDependencyReason::SharedInterface
);
choice!(
    dependency_adaptation,
    DEPENDENCY_ADAPTATION_DESCRIPTOR,
    DEPENDENCY_ADAPTATION_SIGNATURES,
    "fea.SequenceDependencyReason.adaptation",
    "value = fea.SequenceDependencyReason.adaptation()",
    "Order steps across adaptation.",
    DEPENDENCY_REASON_CLASS,
    SequenceDependencyReason::Adaptation
);

choice!(
    transition_reuse,
    TRANSITION_REUSE_DESCRIPTOR,
    TRANSITION_REUSE_SIGNATURES,
    "fea.StateTransitionMethod.reuse",
    "value = fea.StateTransitionMethod.reuse()",
    "Reuse an unchanged state.",
    TRANSITION_METHOD_CLASS,
    StateTransitionMethod::Reuse
);
choice!(
    transition_conservative,
    TRANSITION_CONSERVATIVE_DESCRIPTOR,
    TRANSITION_CONSERVATIVE_SIGNATURES,
    "fea.StateTransitionMethod.conservativeProjection",
    "value = fea.StateTransitionMethod.conservativeProjection()",
    "Project state conservatively.",
    TRANSITION_METHOD_CLASS,
    StateTransitionMethod::ConservativeProjection
);
choice!(
    transition_interpolation,
    TRANSITION_INTERPOLATION_DESCRIPTOR,
    TRANSITION_INTERPOLATION_SIGNATURES,
    "fea.StateTransitionMethod.interpolation",
    "value = fea.StateTransitionMethod.interpolation()",
    "Interpolate state with evidence.",
    TRANSITION_METHOD_CLASS,
    StateTransitionMethod::Interpolation
);
choice!(
    transition_linearize,
    TRANSITION_LINEARIZE_DESCRIPTOR,
    TRANSITION_LINEARIZE_SIGNATURES,
    "fea.StateTransitionMethod.linearize",
    "value = fea.StateTransitionMethod.linearize()",
    "Linearize about an upstream state.",
    TRANSITION_METHOD_CLASS,
    StateTransitionMethod::Linearize
);
choice!(
    transition_operator,
    TRANSITION_OPERATOR_DESCRIPTOR,
    TRANSITION_OPERATOR_SIGNATURES,
    "fea.StateTransitionMethod.deriveOperatorContribution",
    "value = fea.StateTransitionMethod.deriveOperatorContribution()",
    "Derive an operator contribution from state.",
    TRANSITION_METHOD_CLASS,
    StateTransitionMethod::DeriveOperatorContribution
);
choice!(
    transition_guess,
    TRANSITION_GUESS_DESCRIPTOR,
    TRANSITION_GUESS_SIGNATURES,
    "fea.StateTransitionMethod.initialGuessOnly",
    "value = fea.StateTransitionMethod.initialGuessOnly()",
    "Use state only as an initial guess.",
    TRANSITION_METHOD_CLASS,
    StateTransitionMethod::InitialGuessOnly
);

choice!(
    topology_identical,
    TOPOLOGY_IDENTICAL_DESCRIPTOR,
    TOPOLOGY_IDENTICAL_SIGNATURES,
    "fea.TopologyTransitionPolicy.requireIdentical",
    "value = fea.TopologyTransitionPolicy.requireIdentical()",
    "Require identical topology.",
    TOPOLOGY_TRANSITION_CLASS,
    TopologyTransitionPolicy::RequireIdentical
);
choice!(
    topology_project,
    TOPOLOGY_PROJECT_DESCRIPTOR,
    TOPOLOGY_PROJECT_SIGNATURES,
    "fea.TopologyTransitionPolicy.projectWithEvidence",
    "value = fea.TopologyTransitionPolicy.projectWithEvidence()",
    "Permit evidenced topology projection.",
    TOPOLOGY_TRANSITION_CLASS,
    TopologyTransitionPolicy::ProjectWithEvidence
);
choice!(
    topology_adapted,
    TOPOLOGY_ADAPTED_DESCRIPTOR,
    TOPOLOGY_ADAPTED_SIGNATURES,
    "fea.TopologyTransitionPolicy.adaptedMeshLineage",
    "value = fea.TopologyTransitionPolicy.adaptedMeshLineage()",
    "Require an adapted-mesh lineage.",
    TOPOLOGY_TRANSITION_CLASS,
    TopologyTransitionPolicy::AdaptedMeshLineage
);

choice!(
    reference_preserve,
    REFERENCE_PRESERVE_DESCRIPTOR,
    REFERENCE_PRESERVE_SIGNATURES,
    "fea.ReferenceStatePolicy.preserve",
    "value = fea.ReferenceStatePolicy.preserve()",
    "Preserve the producer reference state.",
    REFERENCE_STATE_CLASS,
    ReferenceStatePolicy::Preserve
);
choice!(
    reference_receiving,
    REFERENCE_RECEIVING_DESCRIPTOR,
    REFERENCE_RECEIVING_SIGNATURES,
    "fea.ReferenceStatePolicy.receivingStepOrigin",
    "value = fea.ReferenceStatePolicy.receivingStepOrigin()",
    "Use the receiving step origin.",
    REFERENCE_STATE_CLASS,
    ReferenceStatePolicy::ReceivingStepOrigin
);
choice!(
    reference_linearization,
    REFERENCE_LINEARIZATION_DESCRIPTOR,
    REFERENCE_LINEARIZATION_SIGNATURES,
    "fea.ReferenceStatePolicy.explicitLinearizationPoint",
    "value = fea.ReferenceStatePolicy.explicitLinearizationPoint()",
    "Use an explicit linearization point.",
    REFERENCE_STATE_CLASS,
    ReferenceStatePolicy::ExplicitLinearizationPoint
);

pub(crate) fn state_kind(builtin: &'static str, value: &Value) -> BuiltinResult<StateKind> {
    from_typed_value(builtin, STATE_KIND_CLASS, value)
}
pub(crate) fn state_completeness(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<StateCompleteness> {
    from_typed_value(builtin, STATE_COMPLETENESS_CLASS, value)
}
pub(crate) fn state_persistence(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<StatePersistence> {
    from_typed_value(builtin, STATE_PERSISTENCE_CLASS, value)
}
pub(crate) fn state_use(builtin: &'static str, value: &Value) -> BuiltinResult<StateUse> {
    from_typed_value(builtin, STATE_USE_CLASS, value)
}
pub(crate) fn initial_conflict(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<InitialStateConflictPolicy> {
    from_typed_value(builtin, INITIAL_CONFLICT_CLASS, value)
}
pub(crate) fn dependency_reason(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<SequenceDependencyReason> {
    from_typed_value(builtin, DEPENDENCY_REASON_CLASS, value)
}
pub(crate) fn transition_method(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<StateTransitionMethod> {
    from_typed_value(builtin, TRANSITION_METHOD_CLASS, value)
}
pub(crate) fn topology_transition(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<TopologyTransitionPolicy> {
    from_typed_value(builtin, TOPOLOGY_TRANSITION_CLASS, value)
}
pub(crate) fn reference_state(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<ReferenceStatePolicy> {
    from_typed_value(builtin, REFERENCE_STATE_CLASS, value)
}

pub(super) fn ensure_classes_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        register_immutable_class(
            STATE_KIND_CLASS,
            "runmat.fea.state-kind",
            VALUE_FIELD,
            &[
                ("structuralPrimary", "fea.StateKind.structuralPrimary"),
                ("structuralKinematics", "fea.StateKind.structuralKinematics"),
                (
                    "structuralStressStrain",
                    "fea.StateKind.structuralStressStrain",
                ),
                ("geometricStiffness", "fea.StateKind.geometricStiffness"),
                ("materialHistory", "fea.StateKind.materialHistory"),
                ("contactHistory", "fea.StateKind.contactHistory"),
                ("thermal", "fea.StateKind.thermal"),
                (
                    "electromagneticStatic",
                    "fea.StateKind.electromagneticStatic",
                ),
                (
                    "electromagneticComplex",
                    "fea.StateKind.electromagneticComplex",
                ),
                ("acousticComplex", "fea.StateKind.acousticComplex"),
                ("fluid", "fea.StateKind.fluid"),
                ("movingMesh", "fea.StateKind.movingMesh"),
                ("modalBasis", "fea.StateKind.modalBasis"),
                ("coupledInterface", "fea.StateKind.coupledInterface"),
                ("integrator", "fea.StateKind.integrator"),
            ],
        );
        register_immutable_class(
            STATE_COMPLETENESS_CLASS,
            "runmat.fea.state-completeness",
            VALUE_FIELD,
            &[
                ("complete", "fea.StateCompleteness.complete"),
                ("checkpoint", "fea.StateCompleteness.checkpoint"),
                ("partial", "fea.StateCompleteness.partial"),
            ],
        );
        register_immutable_class(
            STATE_PERSISTENCE_CLASS,
            "runmat.fea.state-persistence",
            VALUE_FIELD,
            &[
                ("canonical", "fea.StatePersistence.canonical"),
                (
                    "untilDependentsComplete",
                    "fea.StatePersistence.untilDependentsComplete",
                ),
                ("checkpointOnly", "fea.StatePersistence.checkpointOnly"),
            ],
        );
        register_immutable_class(
            STATE_USE_CLASS,
            "runmat.fea.state-use",
            VALUE_FIELD,
            &[
                ("physicalInitialState", "fea.StateUse.physicalInitialState"),
                ("linearizationPoint", "fea.StateUse.linearizationPoint"),
                ("prestress", "fea.StateUse.prestress"),
                ("initialGuess", "fea.StateUse.initialGuess"),
                ("modalBasis", "fea.StateUse.modalBasis"),
                ("coupledInterface", "fea.StateUse.coupledInterface"),
                ("restart", "fea.StateUse.restart"),
            ],
        );
        register_immutable_class(
            INITIAL_CONFLICT_CLASS,
            "runmat.fea.initial-state-conflict-policy",
            VALUE_FIELD,
            &[
                ("reject", "fea.InitialStateConflictPolicy.reject"),
                (
                    "replaceInitialCondition",
                    "fea.InitialStateConflictPolicy.replaceInitialCondition",
                ),
                (
                    "augmentDisjointQuantities",
                    "fea.InitialStateConflictPolicy.augmentDisjointQuantities",
                ),
            ],
        );
        register_immutable_class(
            DEPENDENCY_REASON_CLASS,
            "runmat.fea.sequence-dependency-reason",
            VALUE_FIELD,
            &[
                (
                    "stateTransfer",
                    "fea.SequenceDependencyReason.stateTransfer",
                ),
                (
                    "authoredOrdering",
                    "fea.SequenceDependencyReason.authoredOrdering",
                ),
                (
                    "sharedInterface",
                    "fea.SequenceDependencyReason.sharedInterface",
                ),
                ("adaptation", "fea.SequenceDependencyReason.adaptation"),
            ],
        );
        register_immutable_class(
            TRANSITION_METHOD_CLASS,
            "runmat.fea.state-transition-method",
            VALUE_FIELD,
            &[
                ("reuse", "fea.StateTransitionMethod.reuse"),
                (
                    "conservativeProjection",
                    "fea.StateTransitionMethod.conservativeProjection",
                ),
                ("interpolation", "fea.StateTransitionMethod.interpolation"),
                ("linearize", "fea.StateTransitionMethod.linearize"),
                (
                    "deriveOperatorContribution",
                    "fea.StateTransitionMethod.deriveOperatorContribution",
                ),
                (
                    "initialGuessOnly",
                    "fea.StateTransitionMethod.initialGuessOnly",
                ),
            ],
        );
        register_immutable_class(
            TOPOLOGY_TRANSITION_CLASS,
            "runmat.fea.topology-transition-policy",
            VALUE_FIELD,
            &[
                (
                    "requireIdentical",
                    "fea.TopologyTransitionPolicy.requireIdentical",
                ),
                (
                    "projectWithEvidence",
                    "fea.TopologyTransitionPolicy.projectWithEvidence",
                ),
                (
                    "adaptedMeshLineage",
                    "fea.TopologyTransitionPolicy.adaptedMeshLineage",
                ),
            ],
        );
        register_immutable_class(
            REFERENCE_STATE_CLASS,
            "runmat.fea.reference-state-policy",
            VALUE_FIELD,
            &[
                ("preserve", "fea.ReferenceStatePolicy.preserve"),
                (
                    "receivingStepOrigin",
                    "fea.ReferenceStatePolicy.receivingStepOrigin",
                ),
                (
                    "explicitLinearizationPoint",
                    "fea.ReferenceStatePolicy.explicitLinearizationPoint",
                ),
            ],
        );
    });
}
