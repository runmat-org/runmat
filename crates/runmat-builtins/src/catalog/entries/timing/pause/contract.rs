use crate::*;

pub const PAUSE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pause-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pause with an explicit GPU-resident argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PauseGpuInputExtension"),
};
pub const PAUSE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [PAUSE_GPU_INPUT_EXTENSION];

const INTEGER_DURATION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "duration",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Every built-in integer scalar is admitted as a nonnegative real duration.",
    }];
pub const PAUSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "pause(integer_duration)", inputs: &INTEGER_DURATION_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::ScalarOnly,
    notes: "The scalar is validated from authoritative storage before converting to host timer seconds. Explicit resident input is gated; automatic residency gathers.",
}];

const OUTPUT_EMPTY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty array when waiting.",
}];
const OUTPUT_STATE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "state",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Previous or current pause state, 'on' or 'off'.",
}];
const INPUT_NONE: [BuiltinParamDescriptor; 0] = [];
const INPUT_DURATION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "duration",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonnegative scalar duration in seconds, or an empty value for keypress wait.",
}];
const INPUT_COMMAND: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "command",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One of 'on', 'off', or 'query'.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "out = pause()",
        inputs: &INPUT_NONE,
        outputs: &OUTPUT_EMPTY,
    },
    BuiltinSignatureDescriptor {
        label: "out = pause(duration)",
        inputs: &INPUT_DURATION,
        outputs: &OUTPUT_EMPTY,
    },
    BuiltinSignatureDescriptor {
        label: "state = pause(command)",
        inputs: &INPUT_COMMAND,
        outputs: &OUTPUT_STATE,
    },
];
pub const PAUSE_ERROR_INVALID_ARG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAUSE.INVALID_ARG",
    identifier: Some("RunMat:pause:InvalidInputArgument"),
    when: "Input is malformed, unsupported, nonscalar, or a negative/non-finite duration.",
    message: "pause: invalid input argument",
};
pub const PAUSE_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAUSE.TOO_MANY_INPUTS",
    identifier: Some("RunMat:pause:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "pause: too many input arguments",
};
pub const PAUSE_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAUSE.GPU_GATHER_FAILED",
    identifier: Some("RunMat:pause:GpuGatherFailed"),
    when: "Gathering a resident input through its owner fails.",
    message: "pause: failed to gather gpu input",
};
pub const PAUSE_ERROR_CONTEXT_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAUSE.CONTEXT_UNAVAILABLE",
    identifier: Some("RunMat:pause:ContextUnavailable"),
    when: "No invocation timing context is installed.",
    message: "pause: runtime context is unavailable",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    PAUSE_ERROR_INVALID_ARG,
    PAUSE_ERROR_TOO_MANY_INPUTS,
    PAUSE_ERROR_GATHER_FAILED,
    PAUSE_ERROR_CONTEXT_UNAVAILABLE,
];
pub const PAUSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
