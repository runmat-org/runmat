use crate::*;
pub const YSCALE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [AXES_CONTROL_SETTER_OUTPUT_EXTENSION];

const YSCALE_OUTPUT_SCALE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "scale",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current Y-axis scale, 'linear' or 'log'.",
}];

const YSCALE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const YSCALE_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
}];

const YSCALE_INPUTS_SCALE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "scale",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scale string: 'linear' or 'log'.",
}];

const YSCALE_INPUTS_AX_SCALE: [BuiltinParamDescriptor; 2] = [
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

const YSCALE_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const YSCALE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "scale = yscale()",
        inputs: &YSCALE_INPUTS_NONE,
        outputs: &YSCALE_OUTPUT_SCALE,
    },
    BuiltinSignatureDescriptor {
        label: "scale = yscale(ax)",
        inputs: &YSCALE_INPUTS_AX,
        outputs: &YSCALE_OUTPUT_SCALE,
    },
    BuiltinSignatureDescriptor {
        label: "yscale(scale)",
        inputs: &YSCALE_INPUTS_SCALE,
        outputs: &YSCALE_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "yscale(ax, scale)",
        inputs: &YSCALE_INPUTS_AX_SCALE,
        outputs: &YSCALE_OUTPUTS_NONE,
    },
];

const YSCALE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YSCALE.INVALID_ARGUMENT",
    identifier: Some("RunMat:yscale:InvalidArgument"),
    when: "Argument count, scale value, or axes handle is invalid.",
    message: "yscale: invalid argument",
};

const YSCALE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YSCALE.INTERNAL",
    identifier: Some("RunMat:yscale:Internal"),
    when: "Internal plotting state update fails.",
    message: "yscale: internal operation failed",
};

const YSCALE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    YSCALE_ERROR_INVALID_ARGUMENT,
    YSCALE_ERROR_INTERNAL,
    YSCALE_ERROR_TOO_MANY_OUTPUTS,
];

pub const YSCALE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &YSCALE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &YSCALE_ERRORS,
};
pub const YSCALE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YSCALE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:yscale:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "yscale: at most one output is supported",
};
