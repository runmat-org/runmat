//! Signatures, parameters, and integer facts for `fea.StepObjectActivation` constructors.
//! The runtime binding re-exports these facts; it does not redefine them.

use crate::fea_contract::descriptors::ERRORS;
use crate::*;

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

pub const FEA_STEP_OBJECT_ACTIVATION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Typed activation constructors accept identifiers and typed FEA objects only.",
    };

const NEW_ACTIVATION_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "object",
        BuiltinParamType::Any,
        "fea.StepObjectRef naming the object.",
    ),
    required(
        "lifecycle",
        BuiltinParamType::Any,
        "fea.ActivationLifecycle for the object in this step.",
    ),
];
const NEW_ACTIVATION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "activation",
    BuiltinParamType::Any,
    "fea.StepObjectActivation object.",
)];
const NEW_ACTIVATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "activation = fea.StepObjectActivation.new(object, lifecycle)",
    inputs: &NEW_ACTIVATION_INPUTS,
    outputs: &NEW_ACTIVATION_OUTPUTS,
}];
pub const FEA_STEP_OBJECT_ACTIVATION_NEW_ACTIVATION_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &NEW_ACTIVATION_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
