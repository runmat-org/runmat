use crate::*;
pub const GRID_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "Graphics state operation, not numeric array computation; control text and structural graphics identities do not execute integer arithmetic or provider kernels." };
pub const GRID_OUTPUT_ENABLED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enabled",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grid enabled state after command execution.",
}];
pub const GRID_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
pub const GRID_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Grid mode token ('on'|'off'|'minor').",
}];
pub const GRID_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "enabled = grid()",
        inputs: &GRID_INPUTS_NONE,
        outputs: &GRID_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = grid(mode)",
        inputs: &GRID_INPUTS_MODE,
        outputs: &GRID_OUTPUT_ENABLED,
    },
];
pub const GRID_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRID.INVALID_ARGUMENT",
    identifier: Some("RunMat:grid:InvalidArgument"),
    when: "Grid mode argument is unsupported.",
    message: "grid: invalid argument",
};
pub const GRID_ERRORS: [BuiltinErrorDescriptor; 2] =
    [GRID_ERROR_INVALID_ARGUMENT, GRID_ERROR_OUTPUTS];
pub const GRID_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GRID_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GRID_ERRORS,
};
pub const GRID_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRID.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:grid:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "grid: at most one output is supported",
};
pub const GRID_STATE_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor { id: "grid-state-output", mode: BuiltinExtensionMode::RunMatOnly, description: "The assigned logical state/success output is a RunMat extension; MATLAB command calls have no output.", error_identifier: Some("RunMat:compatibility:GRIDStateOutputExtension") };
pub const GRID_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GRID_STATE_OUTPUT_EXTENSION];
