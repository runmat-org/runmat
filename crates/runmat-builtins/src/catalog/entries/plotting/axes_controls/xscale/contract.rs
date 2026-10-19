use crate::*;
pub const XSCALE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [AXES_CONTROL_SETTER_OUTPUT_EXTENSION];

const XSCALE_OUTPUT_SCALE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "scale",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current X-axis scale, 'linear' or 'log'.",
}];

const XSCALE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const XSCALE_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
}];

const XSCALE_INPUTS_SCALE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "scale",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scale string: 'linear' or 'log'.",
}];

const XSCALE_INPUTS_AX_SCALE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "scale",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scale string: 'linear' or 'log'.",
    },
];

const XSCALE_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const XSCALE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "scale = xscale()",
        inputs: &XSCALE_INPUTS_NONE,
        outputs: &XSCALE_OUTPUT_SCALE,
    },
    BuiltinSignatureDescriptor {
        label: "scale = xscale(ax)",
        inputs: &XSCALE_INPUTS_AX,
        outputs: &XSCALE_OUTPUT_SCALE,
    },
    BuiltinSignatureDescriptor {
        label: "xscale(scale)",
        inputs: &XSCALE_INPUTS_SCALE,
        outputs: &XSCALE_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "xscale(ax, scale)",
        inputs: &XSCALE_INPUTS_AX_SCALE,
        outputs: &XSCALE_OUTPUTS_NONE,
    },
];

const XSCALE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XSCALE.INVALID_ARGUMENT",
    identifier: Some("RunMat:xscale:InvalidArgument"),
    when: "Argument count, scale value, or axes handle is invalid.",
    message: "xscale: invalid argument",
};

const XSCALE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XSCALE.INTERNAL",
    identifier: Some("RunMat:xscale:Internal"),
    when: "Internal plotting state update fails.",
    message: "xscale: internal operation failed",
};

const XSCALE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    XSCALE_ERROR_INVALID_ARGUMENT,
    XSCALE_ERROR_INTERNAL,
    XSCALE_ERROR_TOO_MANY_OUTPUTS,
];

pub const XSCALE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XSCALE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XSCALE_ERRORS,
};
pub const XSCALE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XSCALE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:xscale:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "xscale: at most one output is supported",
};
