use std::sync::Once;

use runmat_analysis_core::{
    ActivationLifecycle, ProcedureStepId, StepObjectActivation, StepObjectRef,
};

use super::*;
use crate::builtins::fea::study_values::{
    from_typed_value, nonempty_id, object_or_cell, register_immutable_runtime_classes,
    register_immutable_schemas, to_typed_value, ImmutableStudyClass, VALUE_FIELD,
};

pub(super) const OBJECT_REF_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_STEP_OBJECT_REF_CLASS.display_name();
pub(super) const LIFECYCLE_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_ACTIVATION_LIFECYCLE_CLASS
        .display_name();
pub(super) const ACTIVATION_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_STEP_OBJECT_ACTIVATION_CLASS
        .display_name();

const ACTIVATION_FIELDS: &[&str] = &["object", "lifecycle"];
macro_rules! object_ref_builtin {
    ($function:ident, $builtin:literal, $variant:ident, $descriptor:ident) => {
        #[runtime_builtin(
                                            name = $builtin,
                                            binding_variant = "default",
                                            builtin_path = "crate::builtins::fea::activations"
                                        )]
        pub async fn $function(id: String) -> BuiltinResult<Value> {
            crate::builtins::fea::constraints::fixed_outputs(
                &runmat_builtins::$descriptor,
                $builtin,
            )?;
            to_typed_value(
                $builtin,
                OBJECT_REF_CLASS,
                StepObjectRef::$variant(nonempty_id($builtin, "object id", id)?),
            )
        }
    };
}

object_ref_builtin!(
    constraint,
    "fea.StepObjectRef.constraint",
    Constraint,
    FEA_STEP_OBJECT_REF_CONSTRAINT_DESCRIPTOR
);
object_ref_builtin!(
    boundary,
    "fea.StepObjectRef.boundary",
    Boundary,
    FEA_STEP_OBJECT_REF_BOUNDARY_DESCRIPTOR
);
object_ref_builtin!(
    connector,
    "fea.StepObjectRef.connector",
    Connector,
    FEA_STEP_OBJECT_REF_CONNECTOR_DESCRIPTOR
);
object_ref_builtin!(
    interface,
    "fea.StepObjectRef.interface",
    Interface,
    FEA_STEP_OBJECT_REF_INTERFACE_DESCRIPTOR
);
object_ref_builtin!(
    excitation,
    "fea.StepObjectRef.excitation",
    Excitation,
    FEA_STEP_OBJECT_REF_EXCITATION_DESCRIPTOR
);
object_ref_builtin!(
    initial_condition,
    "fea.StepObjectRef.initialCondition",
    InitialCondition,
    FEA_STEP_OBJECT_REF_INITIAL_CONDITION_DESCRIPTOR
);
object_ref_builtin!(
    material_parameter,
    "fea.StepObjectRef.materialParameter",
    MaterialParameter,
    FEA_STEP_OBJECT_REF_MATERIAL_PARAMETER_DESCRIPTOR
);

#[runtime_builtin(
    name = "fea.ActivationLifecycle.activate",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::activations"
)]
pub async fn activate() -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_ACTIVATION_LIFECYCLE_ACTIVATE_DESCRIPTOR,
        "fea.ActivationLifecycle.activate",
    )?;
    to_typed_value(
        "fea.ActivationLifecycle.activate",
        LIFECYCLE_CLASS,
        ActivationLifecycle::Activate,
    )
}

#[runtime_builtin(
    name = "fea.ActivationLifecycle.continueFrom",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::activations"
)]
pub async fn continue_from(from_step: String) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_ACTIVATION_LIFECYCLE_CONTINUE_FROM_DESCRIPTOR,
        "fea.ActivationLifecycle.continueFrom",
    )?;
    const NAME: &str = "fea.ActivationLifecycle.continueFrom";
    to_typed_value(
        NAME,
        LIFECYCLE_CLASS,
        ActivationLifecycle::Continue {
            from_step: ProcedureStepId(nonempty_id(NAME, "from step", from_step)?),
        },
    )
}

#[runtime_builtin(
    name = "fea.ActivationLifecycle.deactivate",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::activations"
)]
pub async fn deactivate(from_step: String) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_ACTIVATION_LIFECYCLE_DEACTIVATE_DESCRIPTOR,
        "fea.ActivationLifecycle.deactivate",
    )?;
    const NAME: &str = "fea.ActivationLifecycle.deactivate";
    to_typed_value(
        NAME,
        LIFECYCLE_CLASS,
        ActivationLifecycle::Deactivate {
            from_step: ProcedureStepId(nonempty_id(NAME, "from step", from_step)?),
        },
    )
}

#[runtime_builtin(
    name = "fea.ActivationLifecycle.modify",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::activations"
)]
pub async fn modify(prior_step: String, prior_object: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_ACTIVATION_LIFECYCLE_MODIFY_DESCRIPTOR,
        "fea.ActivationLifecycle.modify",
    )?;
    const NAME: &str = "fea.ActivationLifecycle.modify";
    to_typed_value(
        NAME,
        LIFECYCLE_CLASS,
        ActivationLifecycle::Modify {
            prior_step: ProcedureStepId(nonempty_id(NAME, "prior step", prior_step)?),
            prior_object: from_typed_value(NAME, OBJECT_REF_CLASS, &prior_object)?,
        },
    )
}

#[runtime_builtin(
    name = "fea.StepObjectActivation.new",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::activations"
)]
pub async fn activation(object: Value, lifecycle: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_STEP_OBJECT_ACTIVATION_NEW_ACTIVATION_DESCRIPTOR,
        "fea.StepObjectActivation.new",
    )?;
    const NAME: &str = "fea.StepObjectActivation.new";
    serializable_to_object(
        NAME,
        &ERROR_INTERNAL,
        ACTIVATION_CLASS,
        &StepObjectActivation {
            object: from_typed_value(NAME, OBJECT_REF_CLASS, &object)?,
            lifecycle: from_typed_value(NAME, LIFECYCLE_CLASS, &lifecycle)?,
        },
    )
}

pub(super) fn activations(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<StepObjectActivation>> {
    object_or_cell(name, value, ACTIVATION_CLASS)
}

const ACTIVATION_STUDY_CLASSES: [ImmutableStudyClass; 3] = [
    ImmutableStudyClass {
        class_name: OBJECT_REF_CLASS,
        type_identity: "runmat.fea.step-object-ref",
        fields: VALUE_FIELD,
        methods: &[
            ("constraint", "fea.StepObjectRef.constraint"),
            ("boundary", "fea.StepObjectRef.boundary"),
            ("connector", "fea.StepObjectRef.connector"),
            ("interface", "fea.StepObjectRef.interface"),
            ("excitation", "fea.StepObjectRef.excitation"),
            ("initialCondition", "fea.StepObjectRef.initialCondition"),
            ("materialParameter", "fea.StepObjectRef.materialParameter"),
        ],
    },
    ImmutableStudyClass {
        class_name: LIFECYCLE_CLASS,
        type_identity: "runmat.fea.activation-lifecycle",
        fields: VALUE_FIELD,
        methods: &[
            ("activate", "fea.ActivationLifecycle.activate"),
            ("continueFrom", "fea.ActivationLifecycle.continueFrom"),
            ("modify", "fea.ActivationLifecycle.modify"),
            ("deactivate", "fea.ActivationLifecycle.deactivate"),
        ],
    },
    ImmutableStudyClass {
        class_name: ACTIVATION_CLASS,
        type_identity: "runmat.fea.step-object-activation",
        fields: ACTIVATION_FIELDS,
        methods: &[("new", "fea.StepObjectActivation.new")],
    },
];

pub(super) fn ensure_activation_classes_registered() {
    static SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            OBJECT_REF_CLASS,
        ));
    SCHEMAS.call_once(|| {
        static FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        register_immutable_schemas(&FIELDS, &ACTIVATION_STUDY_CLASSES);
    });
    RUNTIME_CLASSES.ensure(|| register_immutable_runtime_classes(&ACTIVATION_STUDY_CLASSES));
}
