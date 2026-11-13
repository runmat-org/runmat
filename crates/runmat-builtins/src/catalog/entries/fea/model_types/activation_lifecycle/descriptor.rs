//! Signatures, parameters, and integer facts for `fea.ActivationLifecycle` constructors.
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

pub const FEA_ACTIVATION_LIFECYCLE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Typed activation constructors accept identifiers and typed FEA objects only.",
    };

const ACTIVATE_INPUTS: [BuiltinParamDescriptor; 0] = [];
const ACTIVATE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "lifecycle",
    BuiltinParamType::Any,
    "fea.ActivationLifecycle object.",
)];
const ACTIVATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "lifecycle = fea.ActivationLifecycle.activate()",
    inputs: &ACTIVATE_INPUTS,
    outputs: &ACTIVATE_OUTPUTS,
}];
pub const FEA_ACTIVATION_LIFECYCLE_ACTIVATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ACTIVATE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const CONTINUE_FROM_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "fromStep",
    BuiltinParamType::StringScalar,
    "Id of the upstream step the object stays active from.",
)];
const CONTINUE_FROM_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "lifecycle",
    BuiltinParamType::Any,
    "fea.ActivationLifecycle object.",
)];
const CONTINUE_FROM_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "lifecycle = fea.ActivationLifecycle.continueFrom(fromStep)",
    inputs: &CONTINUE_FROM_INPUTS,
    outputs: &CONTINUE_FROM_OUTPUTS,
}];
pub const FEA_ACTIVATION_LIFECYCLE_CONTINUE_FROM_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &CONTINUE_FROM_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const DEACTIVATE_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "fromStep",
    BuiltinParamType::StringScalar,
    "Id of the upstream step whose object is removed.",
)];
const DEACTIVATE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "lifecycle",
    BuiltinParamType::Any,
    "fea.ActivationLifecycle object.",
)];
const DEACTIVATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "lifecycle = fea.ActivationLifecycle.deactivate(fromStep)",
    inputs: &DEACTIVATE_INPUTS,
    outputs: &DEACTIVATE_OUTPUTS,
}];
pub const FEA_ACTIVATION_LIFECYCLE_DEACTIVATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DEACTIVATE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const MODIFY_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "priorStep",
        BuiltinParamType::StringScalar,
        "Id of the upstream step where the replaced object was active.",
    ),
    required(
        "priorObject",
        BuiltinParamType::Any,
        "fea.StepObjectRef of the object being replaced.",
    ),
];
const MODIFY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "lifecycle",
    BuiltinParamType::Any,
    "fea.ActivationLifecycle object.",
)];
const MODIFY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "lifecycle = fea.ActivationLifecycle.modify(priorStep, priorObject)",
    inputs: &MODIFY_INPUTS,
    outputs: &MODIFY_OUTPUTS,
}];
pub const FEA_ACTIVATION_LIFECYCLE_MODIFY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MODIFY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
