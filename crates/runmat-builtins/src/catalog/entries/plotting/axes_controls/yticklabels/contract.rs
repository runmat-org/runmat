use crate::*;

pub const YTICKLABELS_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "yticklabels-integer-axes-handle",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Allow a typed-integer alias for an encoded axes handle",
        error_identifier: Some("RunMat:compatibility:YticklabelsIntegerAxesHandleExtension"),
    };
pub const YTICKLABELS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    YTICKLABELS_INTEGER_AXES_EXTENSION,
    AXES_CONTROL_SETTER_OUTPUT_EXTENSION,
];
pub const YTICKLABELS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "Tick labels accept text and categorical containers; integer label payloads are invalid, while a typed-integer encoded axes alias is a separately gated RunMat extension." };

const YTICKLABELS_OUTPUT_LABELS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "labels",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current Y-axis tick labels as a cell array of character vectors.",
}];

const YTICKLABELS_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current Y-axis tick label mode, 'auto' or 'manual'.",
}];

const YTICKLABELS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const YTICKLABELS_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
}];

const YTICKLABELS_INPUTS_LABELS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "labels",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String array or cell array of character vectors for Y-axis tick labels.",
}];

const YTICKLABELS_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tick label mode string: 'auto', 'manual', or 'mode'.",
}];

const YTICKLABELS_INPUTS_AX_LABELS: [BuiltinParamDescriptor; 2] = [
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
        description: "String array or cell array of character vectors for Y-axis tick labels.",
    },
];

const YTICKLABELS_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const YTICKLABELS_INPUTS_AX_MODE: [BuiltinParamDescriptor; 2] = [
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

const YTICKLABELS_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "labels = yticklabels()",
        inputs: &YTICKLABELS_INPUTS_NONE,
        outputs: &YTICKLABELS_OUTPUT_LABELS,
    },
    BuiltinSignatureDescriptor {
        label: "labels = yticklabels(ax)",
        inputs: &YTICKLABELS_INPUTS_AX,
        outputs: &YTICKLABELS_OUTPUT_LABELS,
    },
    BuiltinSignatureDescriptor {
        label: "yticklabels(labels)",
        inputs: &YTICKLABELS_INPUTS_LABELS,
        outputs: &YTICKLABELS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "mode = yticklabels('mode')",
        inputs: &YTICKLABELS_INPUTS_MODE,
        outputs: &YTICKLABELS_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "mode = yticklabels(ax, 'mode')",
        inputs: &YTICKLABELS_INPUTS_AX_MODE,
        outputs: &YTICKLABELS_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "yticklabels('auto'|'manual')",
        inputs: &YTICKLABELS_INPUTS_MODE,
        outputs: &YTICKLABELS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "yticklabels(ax, 'auto'|'manual')",
        inputs: &YTICKLABELS_INPUTS_AX_MODE,
        outputs: &YTICKLABELS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "yticklabels(ax, labels)",
        inputs: &YTICKLABELS_INPUTS_AX_LABELS,
        outputs: &YTICKLABELS_OUTPUTS_NONE,
    },
];

const YTICKLABELS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKLABELS.INVALID_ARGUMENT",
    identifier: Some("RunMat:yticklabels:InvalidArgument"),
    when: "Argument count, tick label values, mode, or axes handle is invalid.",
    message: "yticklabels: invalid argument",
};

const YTICKLABELS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKLABELS.INTERNAL",
    identifier: Some("RunMat:yticklabels:Internal"),
    when: "Internal plotting state update fails.",
    message: "yticklabels: internal operation failed",
};

const YTICKLABELS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    YTICKLABELS_ERROR_INVALID_ARGUMENT,
    YTICKLABELS_ERROR_INTERNAL,
    YTICKLABELS_ERROR_TOO_MANY_OUTPUTS,
];

pub const YTICKLABELS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &YTICKLABELS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &YTICKLABELS_ERRORS,
};
pub const YTICKLABELS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKLABELS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:yticklabels:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "yticklabels: at most one output is supported",
};
