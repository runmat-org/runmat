use crate::*;
const PULSTRAN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pulse train sampled at T.",
}];

const PULSTRAN_INPUTS_FUNCTION: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pulse delays, or an N-by-2 delay/amplitude matrix.",
    },
    BuiltinParamDescriptor {
        name: "FUN",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pulse function name or function handle.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional pulse function parameters.",
    },
];

const PULSTRAN_INPUTS_PROTO: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pulse delays, or an N-by-2 delay/amplitude matrix.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sampled prototype pulse.",
    },
    BuiltinParamDescriptor {
        name: "FS",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Prototype sample rate.",
    },
];

const PULSTRAN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = pulstran(T, D, FUN, P1, ...)",
        inputs: &PULSTRAN_INPUTS_FUNCTION,
        outputs: &PULSTRAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = pulstran(T, D, P, FS)",
        inputs: &PULSTRAN_INPUTS_PROTO,
        outputs: &PULSTRAN_OUTPUT,
    },
];

pub const PULSTRAN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INVALID_INPUT",
    identifier: Some("RunMat:pulstran:InvalidInput"),
    when: "T, D, or sampled prototype inputs are not real numeric arrays.",
    message: "pulstran: expected real numeric input",
};

pub const PULSTRAN_ERROR_INVALID_DELAY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INVALID_DELAY",
    identifier: Some("RunMat:pulstran:InvalidDelay"),
    when: "Delay input is not a vector or N-by-2 delay/amplitude matrix.",
    message: "pulstran: D must be a delay vector or N-by-2 delay/amplitude matrix",
};

pub const PULSTRAN_ERROR_INVALID_PULSE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INVALID_PULSE",
    identifier: Some("RunMat:pulstran:InvalidPulse"),
    when: "Pulse function, its returned samples, or sampled prototype is malformed or cannot cross the exact binary64 waveform boundary.",
    message: "pulstran: invalid pulse specification",
};

pub const PULSTRAN_ERROR_INVALID_PARAMETER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INVALID_PARAMETER",
    identifier: Some("RunMat:pulstran:InvalidParameter"),
    when: "Pulse function parameters or sampled prototype rate are malformed.",
    message: "pulstran: invalid pulse parameter",
};

pub const PULSTRAN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.ARG_COUNT",
    identifier: Some("RunMat:pulstran:ArgCount"),
    when: "Required arguments are missing or too many prototype arguments are provided.",
    message: "pulstran: expected pulstran(T, D, FUN, ...) or pulstran(T, D, P, FS)",
};

pub const PULSTRAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INTERNAL",
    identifier: Some("RunMat:pulstran:InternalError"),
    when: "Internal tensor construction, callback result materialization, or GPU gather fails.",
    message: "pulstran: internal error",
};

const PULSTRAN_ERRORS: [BuiltinErrorDescriptor; 6] = [
    PULSTRAN_ERROR_INVALID_INPUT,
    PULSTRAN_ERROR_INVALID_DELAY,
    PULSTRAN_ERROR_INVALID_PULSE,
    PULSTRAN_ERROR_INVALID_PARAMETER,
    PULSTRAN_ERROR_ARG_COUNT,
    PULSTRAN_ERROR_INTERNAL,
];

pub const PULSTRAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PULSTRAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PULSTRAN_ERRORS,
};
