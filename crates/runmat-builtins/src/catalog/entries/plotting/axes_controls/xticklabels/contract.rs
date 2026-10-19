use crate::*;

pub const XTICKLABELS_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "xticklabels-integer-axes-handle",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Allow a typed-integer alias for an encoded axes handle",
        error_identifier: Some("RunMat:compatibility:XticklabelsIntegerAxesHandleExtension"),
    };
pub const XTICKLABELS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    XTICKLABELS_INTEGER_AXES_EXTENSION,
    AXES_CONTROL_SETTER_OUTPUT_EXTENSION,
];
pub const XTICKLABELS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "Tick labels accept text and categorical containers; integer label payloads are invalid, while a typed-integer encoded axes alias is a separately gated RunMat extension." };

const XTICKLABELS_OUTPUT_LABELS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "labels",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current X-axis tick labels as a cell array of character vectors.",
}];

const XTICKLABELS_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current X-axis tick label mode, 'auto' or 'manual'.",
}];

const XTICKLABELS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const XTICKLABELS_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
}];

const XTICKLABELS_INPUTS_LABELS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "labels",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String array or cell array of character vectors for X-axis tick labels.",
}];

const XTICKLABELS_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tick label mode string: 'auto', 'manual', or 'mode'.",
}];

const XTICKLABELS_INPUTS_AX_LABELS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "labels",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String array or cell array of character vectors for X-axis tick labels.",
    },
];

const XTICKLABELS_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const XTICKLABELS_INPUTS_AX_MODE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Tick mode token.",
    },
];

const XTICKLABELS_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "labels = xticklabels()",
        inputs: &XTICKLABELS_INPUTS_NONE,
        outputs: &XTICKLABELS_OUTPUT_LABELS,
    },
    BuiltinSignatureDescriptor {
        label: "labels = xticklabels(ax)",
        inputs: &XTICKLABELS_INPUTS_AX,
        outputs: &XTICKLABELS_OUTPUT_LABELS,
    },
    BuiltinSignatureDescriptor {
        label: "xticklabels(labels)",
        inputs: &XTICKLABELS_INPUTS_LABELS,
        outputs: &XTICKLABELS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "mode = xticklabels('mode')",
        inputs: &XTICKLABELS_INPUTS_MODE,
        outputs: &XTICKLABELS_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "mode = xticklabels(ax, 'mode')",
        inputs: &XTICKLABELS_INPUTS_AX_MODE,
        outputs: &XTICKLABELS_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "xticklabels('auto'|'manual')",
        inputs: &XTICKLABELS_INPUTS_MODE,
        outputs: &XTICKLABELS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "xticklabels(ax, 'auto'|'manual')",
        inputs: &XTICKLABELS_INPUTS_AX_MODE,
        outputs: &XTICKLABELS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "xticklabels(ax, labels)",
        inputs: &XTICKLABELS_INPUTS_AX_LABELS,
        outputs: &XTICKLABELS_OUTPUTS_NONE,
    },
];

const XTICKLABELS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKLABELS.INVALID_ARGUMENT",
    identifier: Some("RunMat:xticklabels:InvalidArgument"),
    when: "Argument count, tick label values, mode, or axes handle is invalid.",
    message: "xticklabels: invalid argument",
};

const XTICKLABELS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKLABELS.INTERNAL",
    identifier: Some("RunMat:xticklabels:Internal"),
    when: "Internal plotting state update fails.",
    message: "xticklabels: internal operation failed",
};

const XTICKLABELS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    XTICKLABELS_ERROR_INVALID_ARGUMENT,
    XTICKLABELS_ERROR_INTERNAL,
    XTICKLABELS_ERROR_TOO_MANY_OUTPUTS,
];

pub const XTICKLABELS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XTICKLABELS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XTICKLABELS_ERRORS,
};
pub const XTICKLABELS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKLABELS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:xticklabels:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "xticklabels: at most one output is supported",
};
