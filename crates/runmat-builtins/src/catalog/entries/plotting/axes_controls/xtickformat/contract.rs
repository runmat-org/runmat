use crate::*;
pub const XTICKFORMAT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [AXES_CONTROL_SETTER_OUTPUT_EXTENSION];

const OUTPUT_FORMAT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fmt",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current X-axis tick label format.",
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
        label: "fmt = xtickformat()",
        inputs: &INPUTS_NONE,
        outputs: &OUTPUT_FORMAT,
    },
    BuiltinSignatureDescriptor {
        label: "xtickformat(fmt)",
        inputs: &INPUTS_FORMAT,
        outputs: &NO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "fmt = xtickformat(ax)",
        inputs: &INPUTS_AX,
        outputs: &OUTPUT_FORMAT,
    },
    BuiltinSignatureDescriptor {
        label: "xtickformat(ax, fmt)",
        inputs: &INPUTS_AX_FORMAT,
        outputs: &NO_OUTPUTS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKFORMAT.INVALID_ARGUMENT",
    identifier: Some("RunMat:xtickformat:InvalidArgument"),
    when: "Argument count, format value, or axes handle is invalid.",
    message: "xtickformat: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKFORMAT.INTERNAL",
    identifier: Some("RunMat:xtickformat:Internal"),
    when: "Internal plotting state update fails.",
    message: "xtickformat: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ERROR_INVALID_ARGUMENT,
    ERROR_INTERNAL,
    XTICKFORMAT_ERROR_TOO_MANY_OUTPUTS,
];

pub const XTICKFORMAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const XTICKFORMAT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKFORMAT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:xtickformat:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "xtickformat: at most one output is supported",
};
