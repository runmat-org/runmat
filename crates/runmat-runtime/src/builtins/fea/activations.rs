use std::sync::Once;

use runmat_analysis_core::{
    ActivationLifecycle, ProcedureStepId, StepObjectActivation, StepObjectRef,
};

use super::*;
use crate::builtins::fea::study_values::{
    from_typed_value, nonempty_id, object_or_cell, register_immutable_class, to_typed_value,
    VALUE_FIELD,
};

pub(super) const OBJECT_REF_CLASS: &str = "fea.StepObjectRef";
pub(super) const LIFECYCLE_CLASS: &str = "fea.ActivationLifecycle";
pub(super) const ACTIVATION_CLASS: &str = "fea.StepObjectActivation";

const ACTIVATION_FIELDS: &[&str] = &["object", "lifecycle"];
const STRUCTURAL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed activation constructors accept identifiers and typed FEA objects only.",
};

const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

const ID_INPUT: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Authored object identifier.",
)];
const STEP_INPUT: [BuiltinParamDescriptor; 1] = [required(
    "fromStep",
    BuiltinParamType::StringScalar,
    "Upstream procedure-step identifier.",
)];
const MODIFY_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "priorStep",
        BuiltinParamType::StringScalar,
        "Upstream procedure-step identifier.",
    ),
    required(
        "priorObject",
        BuiltinParamType::Any,
        "Typed reference to the object being replaced.",
    ),
];
const ACTIVATION_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "object",
        BuiltinParamType::Any,
        "Typed reference to the object active in this step.",
    ),
    required(
        "lifecycle",
        BuiltinParamType::Any,
        "Typed lifecycle for the object in this step.",
    ),
];
const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];

macro_rules! descriptor {
    ($descriptor:ident, $signatures:ident, $label:literal, $inputs:expr) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: $inputs,
            outputs: &OUT_ANY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

macro_rules! object_ref_builtin {
    ($function:ident, $builtin:literal, $variant:ident, $descriptor:ident) => {
        #[runtime_builtin(
                                    name = $builtin,
                                    category = "fea",
                                    summary = "Create a typed procedure-step object reference.",
                                    keywords = "fea,procedure,step,activation,object,typed",
                                    descriptor($descriptor),
                                    integer_audit(STRUCTURAL_INTEGER_AUDIT),
                                    builtin_path = "crate::builtins::fea::activations"
                                )]
        pub async fn $function(id: String) -> BuiltinResult<Value> {
            to_typed_value(
                $builtin,
                OBJECT_REF_CLASS,
                StepObjectRef::$variant(nonempty_id($builtin, "object id", id)?),
            )
        }
    };
}

descriptor!(
    CONSTRAINT_REF_DESCRIPTOR,
    CONSTRAINT_REF_SIGNATURES,
    "object = fea.StepObjectRef.constraint(id)",
    &ID_INPUT
);
descriptor!(
    BOUNDARY_REF_DESCRIPTOR,
    BOUNDARY_REF_SIGNATURES,
    "object = fea.StepObjectRef.boundary(id)",
    &ID_INPUT
);
descriptor!(
    CONNECTOR_REF_DESCRIPTOR,
    CONNECTOR_REF_SIGNATURES,
    "object = fea.StepObjectRef.connector(id)",
    &ID_INPUT
);
descriptor!(
    INTERFACE_REF_DESCRIPTOR,
    INTERFACE_REF_SIGNATURES,
    "object = fea.StepObjectRef.interface(id)",
    &ID_INPUT
);
descriptor!(
    EXCITATION_REF_DESCRIPTOR,
    EXCITATION_REF_SIGNATURES,
    "object = fea.StepObjectRef.excitation(id)",
    &ID_INPUT
);
descriptor!(
    INITIAL_CONDITION_REF_DESCRIPTOR,
    INITIAL_CONDITION_REF_SIGNATURES,
    "object = fea.StepObjectRef.initialCondition(id)",
    &ID_INPUT
);
descriptor!(
    MATERIAL_PARAMETER_REF_DESCRIPTOR,
    MATERIAL_PARAMETER_REF_SIGNATURES,
    "object = fea.StepObjectRef.materialParameter(id)",
    &ID_INPUT
);
object_ref_builtin!(
    constraint,
    "fea.StepObjectRef.constraint",
    Constraint,
    CONSTRAINT_REF_DESCRIPTOR
);
object_ref_builtin!(
    boundary,
    "fea.StepObjectRef.boundary",
    Boundary,
    BOUNDARY_REF_DESCRIPTOR
);
object_ref_builtin!(
    connector,
    "fea.StepObjectRef.connector",
    Connector,
    CONNECTOR_REF_DESCRIPTOR
);
object_ref_builtin!(
    interface,
    "fea.StepObjectRef.interface",
    Interface,
    INTERFACE_REF_DESCRIPTOR
);
object_ref_builtin!(
    excitation,
    "fea.StepObjectRef.excitation",
    Excitation,
    EXCITATION_REF_DESCRIPTOR
);
object_ref_builtin!(
    initial_condition,
    "fea.StepObjectRef.initialCondition",
    InitialCondition,
    INITIAL_CONDITION_REF_DESCRIPTOR
);
object_ref_builtin!(
    material_parameter,
    "fea.StepObjectRef.materialParameter",
    MaterialParameter,
    MATERIAL_PARAMETER_REF_DESCRIPTOR
);

descriptor!(
    ACTIVATE_DESCRIPTOR,
    ACTIVATE_SIGNATURES,
    "lifecycle = fea.ActivationLifecycle.activate()",
    &NO_INPUTS
);
#[runtime_builtin(
    name = "fea.ActivationLifecycle.activate",
    category = "fea",
    summary = "Activate an authored object in a procedure step.",
    keywords = "fea,procedure,step,activation,lifecycle,typed",
    descriptor(ACTIVATE_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::activations"
)]
pub async fn activate() -> BuiltinResult<Value> {
    to_typed_value(
        "fea.ActivationLifecycle.activate",
        LIFECYCLE_CLASS,
        ActivationLifecycle::Activate,
    )
}

descriptor!(
    CONTINUE_DESCRIPTOR,
    CONTINUE_SIGNATURES,
    "lifecycle = fea.ActivationLifecycle.continueFrom(fromStep)",
    &STEP_INPUT
);
#[runtime_builtin(
    name = "fea.ActivationLifecycle.continueFrom",
    category = "fea",
    summary = "Continue an object that remains active from an upstream step.",
    keywords = "fea,procedure,step,activation,continue,typed",
    descriptor(CONTINUE_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::activations"
)]
pub async fn continue_from(from_step: String) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ActivationLifecycle.continueFrom";
    to_typed_value(
        NAME,
        LIFECYCLE_CLASS,
        ActivationLifecycle::Continue {
            from_step: ProcedureStepId(nonempty_id(NAME, "from step", from_step)?),
        },
    )
}

descriptor!(
    DEACTIVATE_DESCRIPTOR,
    DEACTIVATE_SIGNATURES,
    "lifecycle = fea.ActivationLifecycle.deactivate(fromStep)",
    &STEP_INPUT
);
#[runtime_builtin(
    name = "fea.ActivationLifecycle.deactivate",
    category = "fea",
    summary = "Deactivate an object inherited from an upstream step.",
    keywords = "fea,procedure,step,activation,deactivate,typed",
    descriptor(DEACTIVATE_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::activations"
)]
pub async fn deactivate(from_step: String) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ActivationLifecycle.deactivate";
    to_typed_value(
        NAME,
        LIFECYCLE_CLASS,
        ActivationLifecycle::Deactivate {
            from_step: ProcedureStepId(nonempty_id(NAME, "from step", from_step)?),
        },
    )
}

descriptor!(
    MODIFY_DESCRIPTOR,
    MODIFY_SIGNATURES,
    "lifecycle = fea.ActivationLifecycle.modify(priorStep, priorObject)",
    &MODIFY_INPUTS
);
#[runtime_builtin(
    name = "fea.ActivationLifecycle.modify",
    category = "fea",
    summary = "Replace an upstream object with a newly authored object identity.",
    keywords = "fea,procedure,step,activation,modify,typed",
    descriptor(MODIFY_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::activations"
)]
pub async fn modify(prior_step: String, prior_object: Value) -> BuiltinResult<Value> {
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

descriptor!(
    ACTIVATION_DESCRIPTOR,
    ACTIVATION_SIGNATURES,
    "activation = fea.StepObjectActivation.new(object, lifecycle)",
    &ACTIVATION_INPUTS
);
#[runtime_builtin(
    name = "fea.StepObjectActivation.new",
    category = "fea",
    summary = "Bind an authored object and its lifecycle to a procedure step.",
    keywords = "fea,procedure,step,activation,lifecycle,typed",
    descriptor(ACTIVATION_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::activations"
)]
pub async fn activation(object: Value, lifecycle: Value) -> BuiltinResult<Value> {
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

pub(super) fn ensure_activation_classes_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        register_immutable_class(
            OBJECT_REF_CLASS,
            "runmat.fea.step-object-ref",
            VALUE_FIELD,
            &[
                ("constraint", "fea.StepObjectRef.constraint"),
                ("boundary", "fea.StepObjectRef.boundary"),
                ("connector", "fea.StepObjectRef.connector"),
                ("interface", "fea.StepObjectRef.interface"),
                ("excitation", "fea.StepObjectRef.excitation"),
                ("initialCondition", "fea.StepObjectRef.initialCondition"),
                ("materialParameter", "fea.StepObjectRef.materialParameter"),
            ],
        );
        register_immutable_class(
            LIFECYCLE_CLASS,
            "runmat.fea.activation-lifecycle",
            VALUE_FIELD,
            &[
                ("activate", "fea.ActivationLifecycle.activate"),
                ("continueFrom", "fea.ActivationLifecycle.continueFrom"),
                ("modify", "fea.ActivationLifecycle.modify"),
                ("deactivate", "fea.ActivationLifecycle.deactivate"),
            ],
        );
        register_immutable_class(
            ACTIVATION_CLASS,
            "runmat.fea.step-object-activation",
            ACTIVATION_FIELDS,
            &[("new", "fea.StepObjectActivation.new")],
        );
    });
}
