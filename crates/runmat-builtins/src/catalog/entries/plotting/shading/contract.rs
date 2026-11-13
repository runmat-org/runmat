use crate::*;
pub const SHADING_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "Graphics state operation, not numeric array computation; control text and structural graphics identities do not execute integer arithmetic or provider kernels." };
pub const SHADING_OUTPUT_OK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True on successful shading mode update.",
}];
pub const SHADING_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Shading mode token: 'flat'|'interp'|'faceted'.",
}];
pub const SHADING_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ok = shading(mode)",
    inputs: &SHADING_INPUTS_MODE,
    outputs: &SHADING_OUTPUT_OK,
}];
pub const SHADING_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SHADING.INVALID_ARGUMENT",
    identifier: Some("RunMat:shading:InvalidArgument"),
    when: "Shading mode is missing, non-string, or unsupported.",
    message: "shading: invalid argument",
};
pub const SHADING_ERRORS: [BuiltinErrorDescriptor; 2] =
    [SHADING_ERROR_INVALID_ARGUMENT, SHADING_ERROR_OUTPUTS];
pub const SHADING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SHADING_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SHADING_ERRORS,
};
pub const SHADING_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SHADING.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:shading:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "shading: at most one output is supported",
};
pub const SHADING_STATE_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor { id: "shading-state-output", mode: BuiltinExtensionMode::RunMatOnly, description: "The assigned logical state/success output is a RunMat extension; MATLAB command calls have no output.", error_identifier: Some("RunMat:compatibility:SHADINGStateOutputExtension") };
pub const SHADING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SHADING_STATE_OUTPUT_EXTENSION];
