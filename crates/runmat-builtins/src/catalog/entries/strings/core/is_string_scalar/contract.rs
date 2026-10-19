use crate::*;
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar logical type-test result.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isStringScalar(value)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const IS_STRING_SCALAR_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.IS_STRING_SCALAR.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:isStringScalar:TooManyOutputs"),
        when: "More than one output is requested.",
        message: "isStringScalar: too many output arguments",
    };
pub const IS_STRING_SCALAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[IS_STRING_SCALAR_ERROR_TOO_MANY_OUTPUTS],
};
pub const IS_STRING_SCALAR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Universal type predicate: integer host and resident values return scalar false without payload access or provider calls.",
};
