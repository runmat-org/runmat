use std::sync::Once;

use runmat_analysis_core::{
    InitialStateConflictPolicy, ReferenceStatePolicy, SequenceDependencyReason, StateCompleteness,
    StateKind, StatePersistence, StateTransitionMethod, StateUse, TopologyTransitionPolicy,
};

use super::super::*;
use crate::builtins::fea::study_values::{
    from_typed_value, register_immutable_runtime_classes, register_immutable_schemas,
    to_typed_value, ImmutableStudyClass, VALUE_FIELD,
};
use runmat_builtins::fea_contract::study_types::classes;

pub(super) const STATE_KIND_CLASS: &str = classes::FEA_STATE_KIND_CLASS.display_name();
pub(super) const STATE_COMPLETENESS_CLASS: &str =
    classes::FEA_STATE_COMPLETENESS_CLASS.display_name();
pub(super) const STATE_PERSISTENCE_CLASS: &str =
    classes::FEA_STATE_PERSISTENCE_CLASS.display_name();
pub(super) const STATE_USE_CLASS: &str = classes::FEA_STATE_USE_CLASS.display_name();
pub(super) const INITIAL_CONFLICT_CLASS: &str =
    classes::FEA_INITIAL_STATE_CONFLICT_POLICY_CLASS.display_name();
pub(super) const DEPENDENCY_REASON_CLASS: &str =
    classes::FEA_SEQUENCE_DEPENDENCY_REASON_CLASS.display_name();
pub(super) const TRANSITION_METHOD_CLASS: &str =
    classes::FEA_STATE_TRANSITION_METHOD_CLASS.display_name();
pub(super) const TOPOLOGY_TRANSITION_CLASS: &str =
    classes::FEA_TOPOLOGY_TRANSITION_POLICY_CLASS.display_name();
pub(super) const REFERENCE_STATE_CLASS: &str =
    classes::FEA_REFERENCE_STATE_POLICY_CLASS.display_name();

macro_rules! choice {
    ($function:ident, $builtin:literal, $class:expr, $value:expr $(,)?) => {
        #[runtime_builtin(
                                    name = $builtin,
                                    binding_variant = "default",
                                    builtin_path = "crate::builtins::fea::study_policies::sequence"
                                )]
        pub async fn $function() -> BuiltinResult<Value> {
            to_typed_value($builtin, $class, $value)
        }
    };
}

choice!(
    state_structural_primary,
    "fea.StateKind.structuralPrimary",
    STATE_KIND_CLASS,
    StateKind::StructuralPrimary,
);
choice!(
    state_structural_kinematics,
    "fea.StateKind.structuralKinematics",
    STATE_KIND_CLASS,
    StateKind::StructuralKinematics,
);
choice!(
    state_structural_stress_strain,
    "fea.StateKind.structuralStressStrain",
    STATE_KIND_CLASS,
    StateKind::StructuralStressStrain,
);
choice!(
    state_geometric_stiffness,
    "fea.StateKind.geometricStiffness",
    STATE_KIND_CLASS,
    StateKind::GeometricStiffness,
);
choice!(
    state_material_history,
    "fea.StateKind.materialHistory",
    STATE_KIND_CLASS,
    StateKind::MaterialHistory,
);
choice!(
    state_contact_history,
    "fea.StateKind.contactHistory",
    STATE_KIND_CLASS,
    StateKind::ContactHistory,
);
choice!(
    state_thermal,
    "fea.StateKind.thermal",
    STATE_KIND_CLASS,
    StateKind::Thermal,
);
choice!(
    state_electromagnetic_static,
    "fea.StateKind.electromagneticStatic",
    STATE_KIND_CLASS,
    StateKind::ElectromagneticStatic,
);
choice!(
    state_electromagnetic_complex,
    "fea.StateKind.electromagneticComplex",
    STATE_KIND_CLASS,
    StateKind::ElectromagneticComplex,
);
choice!(
    state_acoustic_complex,
    "fea.StateKind.acousticComplex",
    STATE_KIND_CLASS,
    StateKind::AcousticComplex,
);
choice!(
    state_fluid,
    "fea.StateKind.fluid",
    STATE_KIND_CLASS,
    StateKind::Fluid,
);
choice!(
    state_moving_mesh,
    "fea.StateKind.movingMesh",
    STATE_KIND_CLASS,
    StateKind::MovingMesh,
);
choice!(
    state_modal_basis,
    "fea.StateKind.modalBasis",
    STATE_KIND_CLASS,
    StateKind::ModalBasis,
);
choice!(
    state_coupled_interface,
    "fea.StateKind.coupledInterface",
    STATE_KIND_CLASS,
    StateKind::CoupledInterface,
);
choice!(
    state_integrator,
    "fea.StateKind.integrator",
    STATE_KIND_CLASS,
    StateKind::Integrator,
);

choice!(
    completeness_complete,
    "fea.StateCompleteness.complete",
    STATE_COMPLETENESS_CLASS,
    StateCompleteness::Complete,
);
choice!(
    completeness_checkpoint,
    "fea.StateCompleteness.checkpoint",
    STATE_COMPLETENESS_CLASS,
    StateCompleteness::Checkpoint,
);
choice!(
    completeness_partial,
    "fea.StateCompleteness.partial",
    STATE_COMPLETENESS_CLASS,
    StateCompleteness::Partial,
);

choice!(
    persistence_canonical,
    "fea.StatePersistence.canonical",
    STATE_PERSISTENCE_CLASS,
    StatePersistence::Canonical,
);
choice!(
    persistence_until_dependents,
    "fea.StatePersistence.untilDependentsComplete",
    STATE_PERSISTENCE_CLASS,
    StatePersistence::UntilDependentsComplete,
);
choice!(
    persistence_checkpoint_only,
    "fea.StatePersistence.checkpointOnly",
    STATE_PERSISTENCE_CLASS,
    StatePersistence::CheckpointOnly,
);

choice!(
    use_physical_initial,
    "fea.StateUse.physicalInitialState",
    STATE_USE_CLASS,
    StateUse::PhysicalInitialState,
);
choice!(
    use_linearization,
    "fea.StateUse.linearizationPoint",
    STATE_USE_CLASS,
    StateUse::LinearizationPoint,
);
choice!(
    use_prestress,
    "fea.StateUse.prestress",
    STATE_USE_CLASS,
    StateUse::Prestress,
);
choice!(
    use_initial_guess,
    "fea.StateUse.initialGuess",
    STATE_USE_CLASS,
    StateUse::InitialGuess,
);
choice!(
    use_modal_basis,
    "fea.StateUse.modalBasis",
    STATE_USE_CLASS,
    StateUse::ModalBasis,
);
choice!(
    use_coupled_interface,
    "fea.StateUse.coupledInterface",
    STATE_USE_CLASS,
    StateUse::CoupledInterface,
);
choice!(
    use_restart,
    "fea.StateUse.restart",
    STATE_USE_CLASS,
    StateUse::Restart,
);

choice!(
    conflict_reject,
    "fea.InitialStateConflictPolicy.reject",
    INITIAL_CONFLICT_CLASS,
    InitialStateConflictPolicy::Reject,
);
choice!(
    conflict_replace,
    "fea.InitialStateConflictPolicy.replaceInitialCondition",
    INITIAL_CONFLICT_CLASS,
    InitialStateConflictPolicy::ReplaceInitialCondition,
);
choice!(
    conflict_augment,
    "fea.InitialStateConflictPolicy.augmentDisjointQuantities",
    INITIAL_CONFLICT_CLASS,
    InitialStateConflictPolicy::AugmentDisjointQuantities,
);

choice!(
    dependency_state_transfer,
    "fea.SequenceDependencyReason.stateTransfer",
    DEPENDENCY_REASON_CLASS,
    SequenceDependencyReason::StateTransfer,
);
choice!(
    dependency_authored,
    "fea.SequenceDependencyReason.authoredOrdering",
    DEPENDENCY_REASON_CLASS,
    SequenceDependencyReason::AuthoredOrdering,
);
choice!(
    dependency_interface,
    "fea.SequenceDependencyReason.sharedInterface",
    DEPENDENCY_REASON_CLASS,
    SequenceDependencyReason::SharedInterface,
);
choice!(
    dependency_adaptation,
    "fea.SequenceDependencyReason.adaptation",
    DEPENDENCY_REASON_CLASS,
    SequenceDependencyReason::Adaptation,
);

choice!(
    transition_reuse,
    "fea.StateTransitionMethod.reuse",
    TRANSITION_METHOD_CLASS,
    StateTransitionMethod::Reuse,
);
choice!(
    transition_conservative,
    "fea.StateTransitionMethod.conservativeProjection",
    TRANSITION_METHOD_CLASS,
    StateTransitionMethod::ConservativeProjection,
);
choice!(
    transition_interpolation,
    "fea.StateTransitionMethod.interpolation",
    TRANSITION_METHOD_CLASS,
    StateTransitionMethod::Interpolation,
);
choice!(
    transition_linearize,
    "fea.StateTransitionMethod.linearize",
    TRANSITION_METHOD_CLASS,
    StateTransitionMethod::Linearize,
);
choice!(
    transition_operator,
    "fea.StateTransitionMethod.deriveOperatorContribution",
    TRANSITION_METHOD_CLASS,
    StateTransitionMethod::DeriveOperatorContribution,
);
choice!(
    transition_guess,
    "fea.StateTransitionMethod.initialGuessOnly",
    TRANSITION_METHOD_CLASS,
    StateTransitionMethod::InitialGuessOnly,
);

choice!(
    topology_identical,
    "fea.TopologyTransitionPolicy.requireIdentical",
    TOPOLOGY_TRANSITION_CLASS,
    TopologyTransitionPolicy::RequireIdentical,
);
choice!(
    topology_project,
    "fea.TopologyTransitionPolicy.projectWithEvidence",
    TOPOLOGY_TRANSITION_CLASS,
    TopologyTransitionPolicy::ProjectWithEvidence,
);
choice!(
    topology_adapted,
    "fea.TopologyTransitionPolicy.adaptedMeshLineage",
    TOPOLOGY_TRANSITION_CLASS,
    TopologyTransitionPolicy::AdaptedMeshLineage,
);

choice!(
    reference_preserve,
    "fea.ReferenceStatePolicy.preserve",
    REFERENCE_STATE_CLASS,
    ReferenceStatePolicy::Preserve,
);
choice!(
    reference_receiving,
    "fea.ReferenceStatePolicy.receivingStepOrigin",
    REFERENCE_STATE_CLASS,
    ReferenceStatePolicy::ReceivingStepOrigin,
);
choice!(
    reference_linearization,
    "fea.ReferenceStatePolicy.explicitLinearizationPoint",
    REFERENCE_STATE_CLASS,
    ReferenceStatePolicy::ExplicitLinearizationPoint,
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

const SEQUENCE_POLICY_CLASSES: [ImmutableStudyClass; 9] = [
    ImmutableStudyClass {
        class_name: STATE_KIND_CLASS,
        type_identity: "runmat.fea.state-kind",
        fields: VALUE_FIELD,
        methods: &[
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
    },
    ImmutableStudyClass {
        class_name: STATE_COMPLETENESS_CLASS,
        type_identity: "runmat.fea.state-completeness",
        fields: VALUE_FIELD,
        methods: &[
            ("complete", "fea.StateCompleteness.complete"),
            ("checkpoint", "fea.StateCompleteness.checkpoint"),
            ("partial", "fea.StateCompleteness.partial"),
        ],
    },
    ImmutableStudyClass {
        class_name: STATE_PERSISTENCE_CLASS,
        type_identity: "runmat.fea.state-persistence",
        fields: VALUE_FIELD,
        methods: &[
            ("canonical", "fea.StatePersistence.canonical"),
            (
                "untilDependentsComplete",
                "fea.StatePersistence.untilDependentsComplete",
            ),
            ("checkpointOnly", "fea.StatePersistence.checkpointOnly"),
        ],
    },
    ImmutableStudyClass {
        class_name: STATE_USE_CLASS,
        type_identity: "runmat.fea.state-use",
        fields: VALUE_FIELD,
        methods: &[
            ("physicalInitialState", "fea.StateUse.physicalInitialState"),
            ("linearizationPoint", "fea.StateUse.linearizationPoint"),
            ("prestress", "fea.StateUse.prestress"),
            ("initialGuess", "fea.StateUse.initialGuess"),
            ("modalBasis", "fea.StateUse.modalBasis"),
            ("coupledInterface", "fea.StateUse.coupledInterface"),
            ("restart", "fea.StateUse.restart"),
        ],
    },
    ImmutableStudyClass {
        class_name: INITIAL_CONFLICT_CLASS,
        type_identity: "runmat.fea.initial-state-conflict-policy",
        fields: VALUE_FIELD,
        methods: &[
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
    },
    ImmutableStudyClass {
        class_name: DEPENDENCY_REASON_CLASS,
        type_identity: "runmat.fea.sequence-dependency-reason",
        fields: VALUE_FIELD,
        methods: &[
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
    },
    ImmutableStudyClass {
        class_name: TRANSITION_METHOD_CLASS,
        type_identity: "runmat.fea.state-transition-method",
        fields: VALUE_FIELD,
        methods: &[
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
    },
    ImmutableStudyClass {
        class_name: TOPOLOGY_TRANSITION_CLASS,
        type_identity: "runmat.fea.topology-transition-policy",
        fields: VALUE_FIELD,
        methods: &[
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
    },
    ImmutableStudyClass {
        class_name: REFERENCE_STATE_CLASS,
        type_identity: "runmat.fea.reference-state-policy",
        fields: VALUE_FIELD,
        methods: &[
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
    },
];

pub(super) fn ensure_classes_registered() {
    static SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            STATE_KIND_CLASS,
        ));
    SCHEMAS.call_once(|| {
        static FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        register_immutable_schemas(&FIELDS, &SEQUENCE_POLICY_CLASSES);
    });
    RUNTIME_CLASSES.ensure(|| register_immutable_runtime_classes(&SEQUENCE_POLICY_CLASSES));
}

#[cfg(test)]
mod tests;
