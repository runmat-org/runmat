use crate::*;
pub const FUNC2STR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function name string.",
}];

pub const FUNC2STR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fh",
    ty: BuiltinParamType::Callable,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle value.",
}];

pub const FUNC2STR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "name = func2str(fh)",
    inputs: &FUNC2STR_INPUTS,
    outputs: &FUNC2STR_OUTPUT,
}];

pub const FUNC2STR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FUNC2STR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FUNC2STR_ERRORS,
};

pub const FUNC2STR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "func2str accepts only function-handle values and returns handle text; integer, numeric, and provider-resident values reject before provider access.",
    };

pub const FUNC2STR_ERROR_HANDLE_TYPE_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FUNC2STR.HANDLE_TYPE_INVALID",
    identifier: Some("RunMat:Func2StrHandleTypeInvalid"),
    when: "Input is not a supported function handle value.",
    message: "func2str: expected function handle",
};

pub const FUNC2STR_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FUNC2STR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:func2str:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "func2str: too many output arguments",
};
pub const FUNC2STR_ERRORS: [BuiltinErrorDescriptor; 2] = [
    FUNC2STR_ERROR_HANDLE_TYPE_INVALID,
    FUNC2STR_ERROR_TOO_MANY_OUTPUTS,
];
