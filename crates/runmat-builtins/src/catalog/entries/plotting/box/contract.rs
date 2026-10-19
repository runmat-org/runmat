use crate::*;
pub const BOX_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "Graphics state operation, not numeric array computation; control text and structural graphics identities do not execute integer arithmetic or provider kernels." };
pub const BOX_OUTPUT_ENABLED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enabled",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Box outline enabled state after command execution.",
}];
pub const BOX_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
pub const BOX_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Box mode token ('on'|'off').",
}];
pub const BOX_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "enabled = box()",
        inputs: &BOX_INPUTS_NONE,
        outputs: &BOX_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = box(mode)",
        inputs: &BOX_INPUTS_MODE,
        outputs: &BOX_OUTPUT_ENABLED,
    },
];
pub const BOX_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BOX.INVALID_ARGUMENT",
    identifier: Some("RunMat:box:InvalidArgument"),
    when: "Box mode argument is unsupported.",
    message: "box: invalid argument",
};
pub const BOX_ERRORS: [BuiltinErrorDescriptor; 2] = [BOX_ERROR_INVALID_ARGUMENT, BOX_ERROR_OUTPUTS];
pub const BOX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BOX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BOX_ERRORS,
};
pub const BOX_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BOX.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:box:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "box: at most one output is supported",
};
pub const BOX_STATE_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor { id: "box-state-output", mode: BuiltinExtensionMode::RunMatOnly, description: "The assigned logical state/success output is a RunMat extension; MATLAB command calls have no output.", error_identifier: Some("RunMat:compatibility:BOXStateOutputExtension") };
pub const BOX_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [BOX_STATE_OUTPUT_EXTENSION];
