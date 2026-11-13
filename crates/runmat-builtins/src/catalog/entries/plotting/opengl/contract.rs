use crate::*;
const OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Renderer status string or information struct.",
}];

const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const INPUTS_COMMANDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "commands",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Commands such as info, data, hardware, hardwarebasic, software, save, autoselect, or neverselect.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "status = opengl()",
        inputs: &INPUTS_NONE,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "info = opengl('info')",
        inputs: &INPUTS_COMMANDS,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "status = opengl(command, ...)",
        inputs: &INPUTS_COMMANDS,
        outputs: &OUTPUT_VALUE,
    },
];

pub const OPENGL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENGL.INVALID_ARGUMENT",
    identifier: Some("RunMat:opengl:InvalidArgument"),
    when: "Command text or command combination is invalid.",
    message: "opengl: invalid argument",
};

const ERRORS: [BuiltinErrorDescriptor; 2] =
    [OPENGL_ERROR_INVALID_ARGUMENT, OPENGL_ERROR_TOO_MANY_OUTPUTS];

pub const OPENGL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const OPENGL_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENGL.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:opengl:TooManyOutputs"),
    when: "More than 1 output arguments are requested.",
    message: "opengl: too many output arguments",
};
