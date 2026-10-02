use crate::*;
const FILTFILT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero-phase filtered signal.",
}];

const FILTFILT_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal vector.",
    },
];

const FILTFILT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "y = filtfilt(b, a, x)",
    inputs: &FILTFILT_INPUTS,
    outputs: &FILTFILT_OUTPUT,
}];

pub const FILTFILT_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTFILT.ARG_COUNT",
    identifier: Some("RunMat:filtfilt:ArgCount"),
    when: "The argument count is not exactly three.",
    message: "filtfilt: expected filtfilt(b, a, x)",
};

pub const FILTFILT_ERROR_INVALID_COEFFICIENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTFILT.INVALID_COEFFICIENTS",
    identifier: Some("RunMat:filtfilt:InvalidCoefficients"),
    when: "Coefficient inputs are empty, non-vector, or have a zero leading denominator.",
    message: "filtfilt: invalid coefficient input",
};

pub const FILTFILT_ERROR_INVALID_SIGNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTFILT.INVALID_SIGNAL",
    identifier: Some("RunMat:filtfilt:InvalidSignal"),
    when: "The signal is not a numeric/logical vector or is too short for edge reflection.",
    message: "filtfilt: invalid signal input",
};

pub const FILTFILT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTFILT.INTERNAL",
    identifier: Some("RunMat:filtfilt:Internal"),
    when: "Internal state solving, filtering, or tensor construction fails.",
    message: "filtfilt: internal error",
};

pub const FILTFILT_ERROR_PASS_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTFILT.PASS_FAILED",
    identifier: Some("RunMat:filtfilt:PassFailed"),
    when: "A forward or reverse pass fails during preparation, provider execution, or readback.",
    message: "filtfilt: filtering pass failed",
};

pub const FILTFILT_ERROR_PROVIDER_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTFILT.PROVIDER_OUTPUT",
    identifier: Some("RunMat:filtfilt:InvalidProviderOutput"),
    when: "The provider returns a stale, foreign, wrong-class, wrong-shape, or borrowed output allocation.",
    message: "filtfilt: invalid provider result",
};

const FILTFILT_ERRORS: [BuiltinErrorDescriptor; 6] = [
    FILTFILT_ERROR_ARG_COUNT,
    FILTFILT_ERROR_INVALID_COEFFICIENTS,
    FILTFILT_ERROR_INVALID_SIGNAL,
    FILTFILT_ERROR_INTERNAL,
    FILTFILT_ERROR_PASS_FAILED,
    FILTFILT_ERROR_PROVIDER_OUTPUT,
];

pub const FILTFILT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FILTFILT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FILTFILT_ERRORS,
};
