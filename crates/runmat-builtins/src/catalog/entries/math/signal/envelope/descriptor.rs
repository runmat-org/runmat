use crate::*;

const UPPER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "yupper",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Upper signal envelope.",
};
const LOWER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ylower",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Lower signal envelope.",
};
const OUTPUT_UPPER: [BuiltinParamDescriptor; 1] = [UPPER];
const OUTPUT_BOTH: [BuiltinParamDescriptor; 2] = [UPPER, LOWER];
const SIGNAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real-valued vector or matrix, processed column-wise.",
};
const LENGTH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filter/window length or peak separation.",
};
const METHOD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "method",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"analytic\""),
    description: "analytic, rms, or peak.",
};
const CORE: [BuiltinParamDescriptor; 1] = [SIGNAL];
const WITH_LENGTH: [BuiltinParamDescriptor; 2] = [SIGNAL, LENGTH];
const WITH_METHOD: [BuiltinParamDescriptor; 3] = [SIGNAL, LENGTH, METHOD];
const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "yupper = envelope(x)",
        inputs: &CORE,
        outputs: &OUTPUT_UPPER,
    },
    BuiltinSignatureDescriptor {
        label: "[yupper, ylower] = envelope(x)",
        inputs: &CORE,
        outputs: &OUTPUT_BOTH,
    },
    BuiltinSignatureDescriptor {
        label: "yupper = envelope(x, n)",
        inputs: &WITH_LENGTH,
        outputs: &OUTPUT_UPPER,
    },
    BuiltinSignatureDescriptor {
        label: "[yupper, ylower] = envelope(x, n)",
        inputs: &WITH_LENGTH,
        outputs: &OUTPUT_BOTH,
    },
    BuiltinSignatureDescriptor {
        label: "yupper = envelope(x, n, method)",
        inputs: &WITH_METHOD,
        outputs: &OUTPUT_UPPER,
    },
    BuiltinSignatureDescriptor {
        label: "[yupper, ylower] = envelope(x, n, method)",
        inputs: &WITH_METHOD,
        outputs: &OUTPUT_BOTH,
    },
];

pub const ENVELOPE_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENVELOPE.ARG_COUNT",
    identifier: Some("RunMat:envelope:ArgCount"),
    when: "The argument count is outside supported forms.",
    message: "envelope: expected envelope(x), envelope(x, n), or envelope(x, n, method)",
};
pub const ENVELOPE_ERROR_INVALID_SIGNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENVELOPE.INVALID_SIGNAL",
    identifier: Some("RunMat:envelope:InvalidSignal"),
    when: "Input is not a finite real numeric vector or matrix.",
    message: "envelope: x must be a finite real numeric vector or matrix",
};
pub const ENVELOPE_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENVELOPE.INVALID_LENGTH",
    identifier: Some("RunMat:envelope:InvalidLength"),
    when: "Filter/window length or peak separation is not a positive integer.",
    message: "envelope: n must be a positive integer scalar",
};
pub const ENVELOPE_ERROR_INVALID_METHOD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENVELOPE.INVALID_METHOD",
    identifier: Some("RunMat:envelope:InvalidMethod"),
    when: "Method is not analytic, rms, or peak.",
    message: "envelope: method must be 'analytic', 'rms', or 'peak'",
};
pub const ENVELOPE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENVELOPE.INTERNAL",
    identifier: Some("RunMat:envelope:Internal"),
    when: "Envelope computation or output tensor assembly fails internally.",
    message: "envelope: internal error",
};
pub const ENVELOPE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENVELOPE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:envelope:TooManyOutputs"),
    when: "More than two output arguments are requested.",
    message: "envelope: expected at most two output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    ENVELOPE_ERROR_ARG_COUNT,
    ENVELOPE_ERROR_INVALID_SIGNAL,
    ENVELOPE_ERROR_INVALID_LENGTH,
    ENVELOPE_ERROR_INVALID_METHOD,
    ENVELOPE_ERROR_INTERNAL,
    ENVELOPE_ERROR_TOO_MANY_OUTPUTS,
];
pub const ENVELOPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
