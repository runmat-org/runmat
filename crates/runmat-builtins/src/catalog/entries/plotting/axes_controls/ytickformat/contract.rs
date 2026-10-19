use crate::*;
pub const YTICKFORMAT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [AXES_CONTROL_SETTER_OUTPUT_EXTENSION];

const OUTPUT_FORMAT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fmt",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current Y-axis tick label format.",
}];
const NO_OUTPUTS: [BuiltinParamDescriptor; 0] = [];

const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const INPUTS_FORMAT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fmt",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, date, or duration tick label format.",
}];

const INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle or array of axes handles.",
}];

const INPUTS_AX_FORMAT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle or array of axes handles.",
    },
    BuiltinParamDescriptor {
        name: "fmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, date, or duration tick label format.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "fmt = ytickformat()",
        inputs: &INPUTS_NONE,
        outputs: &OUTPUT_FORMAT,
    },
    BuiltinSignatureDescriptor {
        label: "ytickformat(fmt)",
        inputs: &INPUTS_FORMAT,
        outputs: &NO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "fmt = ytickformat(ax)",
        inputs: &INPUTS_AX,
        outputs: &OUTPUT_FORMAT,
    },
    BuiltinSignatureDescriptor {
        label: "ytickformat(ax, fmt)",
        inputs: &INPUTS_AX_FORMAT,
        outputs: &NO_OUTPUTS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKFORMAT.INVALID_ARGUMENT",
    identifier: Some("RunMat:ytickformat:InvalidArgument"),
    when: "Argument count, format value, or axes handle is invalid.",
    message: "ytickformat: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKFORMAT.INTERNAL",
    identifier: Some("RunMat:ytickformat:Internal"),
    when: "Internal plotting state update fails.",
    message: "ytickformat: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ERROR_INVALID_ARGUMENT,
    ERROR_INTERNAL,
    YTICKFORMAT_ERROR_TOO_MANY_OUTPUTS,
];

pub const YTICKFORMAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const YTICKFORMAT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKFORMAT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:ytickformat:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "ytickformat: at most one output is supported",
};
