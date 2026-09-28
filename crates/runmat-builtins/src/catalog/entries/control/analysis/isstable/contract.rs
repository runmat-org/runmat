use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "True when every pole lies inside the stability region for the model's time domain.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO transfer-function model.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isstable(sys)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const ISSTABLE_ERROR_INVALID_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSTABLE.INVALID_MODEL",
    identifier: Some("RunMat:isstable:InvalidModel"),
    when: "Input system is not a valid SISO tf object.",
    message: "isstable: invalid model",
};
pub const ISSTABLE_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSTABLE.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:isstable:UnsupportedModel"),
    when: "Model form is unsupported.",
    message: "isstable: unsupported model",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ISSTABLE_ERROR_INVALID_MODEL,
    ISSTABLE_ERROR_UNSUPPORTED_MODEL,
    BuiltinErrorDescriptor {
        code: "RM.ISSTABLE.INTERNAL",
        identifier: Some("RunMat:isstable:Internal"),
        when: "Pole calculation failed.",
        message: "isstable: internal error",
    },
];

pub const ISSTABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ISSTABLE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "isstable requires a dynamic-system model; fundamental integer host or resident values are invalid-domain inputs and reject before payload or provider access.",
};
