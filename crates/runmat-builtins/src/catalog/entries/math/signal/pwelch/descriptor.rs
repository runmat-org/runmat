use crate::*;

const INPUTS: [BuiltinParamDescriptor; 7] = [
    BuiltinParamDescriptor { name: "x", ty: BuiltinParamType::NumericArray, arity: BuiltinParamArity::Required, default: None, description: "Signal vector or 2-D matrix; each column is processed separately." },
    BuiltinParamDescriptor { name: "window", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Optional, default: Some("[]"), description: "Finite real window vector or positive integer segment length." },
    BuiltinParamDescriptor { name: "noverlap", ty: BuiltinParamType::NumericScalar, arity: BuiltinParamArity::Optional, default: Some("[]"), description: "Nonnegative overlap count smaller than the window length." },
    BuiltinParamDescriptor { name: "nfft", ty: BuiltinParamType::NumericArray, arity: BuiltinParamArity::Optional, default: Some("[]"), description: "DFT length or explicit real frequency vector." },
    BuiltinParamDescriptor { name: "fs", ty: BuiltinParamType::NumericScalar, arity: BuiltinParamArity::Optional, default: None, description: "Positive sample frequency in Hz." },
    BuiltinParamDescriptor { name: "options", ty: BuiltinParamType::StringScalar, arity: BuiltinParamArity::Variadic, default: None, description: "Up to three range (onesided, twosided, centered), scale (psd, power), or trace (mean, maxhold, minhold) selectors in source order." },
    BuiltinParamDescriptor { name: "Parent", ty: BuiltinParamType::AxesHandle, arity: BuiltinParamArity::Optional, default: None, description: "Optional name-value target: an existing RunMat axes handle. UIAxes and Panel containers require graphics identities not yet exposed by RunMat." },
];
const PSD: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pxx",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Welch power spectral estimate in the signal's floating precision.",
}];
const PSD_AND_F: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "pxx",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Welch power spectral estimate in the signal's floating precision.",
    },
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Frequency vector in the signal's floating precision, radians/sample or Hz with fs.",
    },
];
const NO_OUTPUTS: [BuiltinParamDescriptor; 0] = [];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "pxx = pwelch(x, window, noverlap, nfft, fs, options)",
        inputs: &INPUTS,
        outputs: &PSD,
    },
    BuiltinSignatureDescriptor {
        label: "[pxx, f] = pwelch(x, window, noverlap, nfft, fs, options)",
        inputs: &INPUTS,
        outputs: &PSD_AND_F,
    },
    BuiltinSignatureDescriptor {
        label: "pwelch(x, window, noverlap, nfft, fs, options, Parent=ax)",
        inputs: &INPUTS,
        outputs: &NO_OUTPUTS,
    },
];

pub const PWELCH_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.ARG_COUNT",
    identifier: Some("RunMat:pwelch:ArgCount"),
    when: "The argument count is outside supported forms.",
    message:
        "pwelch: expected pwelch(x, [window, [noverlap, [nfft_or_f, [fs, [freqrange_or_scale]]]]])",
};
pub const PWELCH_ERROR_INVALID_SIGNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INVALID_SIGNAL",
    identifier: Some("RunMat:pwelch:InvalidSignal"),
    when: "Input signal is not a numeric vector or matrix.",
    message: "pwelch: x must be a nonempty numeric vector or 2-D matrix",
};
pub const PWELCH_ERROR_INVALID_WINDOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INVALID_WINDOW",
    identifier: Some("RunMat:pwelch:InvalidWindow"),
    when: "Window input is invalid.",
    message: "pwelch: window must be a nonempty vector or positive integer length",
};
pub const PWELCH_ERROR_INVALID_OVERLAP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INVALID_OVERLAP",
    identifier: Some("RunMat:pwelch:InvalidOverlap"),
    when: "Overlap is invalid.",
    message: "pwelch: noverlap must be a nonnegative integer less than window length",
};
pub const PWELCH_ERROR_INVALID_NFFT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INVALID_NFFT",
    identifier: Some("RunMat:pwelch:InvalidNfft"),
    when: "FFT length is invalid.",
    message: "pwelch: nfft must be a positive integer",
};
pub const PWELCH_ERROR_INVALID_FS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INVALID_FS",
    identifier: Some("RunMat:pwelch:InvalidFs"),
    when: "Sampling frequency is invalid.",
    message: "pwelch: fs must be a positive finite scalar",
};
pub const PWELCH_ERROR_INVALID_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INVALID_RANGE",
    identifier: Some("RunMat:pwelch:InvalidRange"),
    when: "Frequency range, spectrum, or trace selector is invalid.",
    message: "pwelch: options must select a frequency range, spectrum type, or trace mode",
};
pub const PWELCH_ERROR_INVALID_PARENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INVALID_PARENT",
    identifier: Some("RunMat:pwelch:InvalidParent"),
    when: "Parent is missing or is not an existing axes handle.",
    message: "pwelch: Parent must be an existing axes handle",
};
pub const PWELCH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INTERNAL",
    identifier: Some("RunMat:pwelch:Internal"),
    when: "Output construction or provider execution fails.",
    message: "pwelch: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 9] = [
    PWELCH_ERROR_ARG_COUNT,
    PWELCH_ERROR_INVALID_SIGNAL,
    PWELCH_ERROR_INVALID_WINDOW,
    PWELCH_ERROR_INVALID_OVERLAP,
    PWELCH_ERROR_INVALID_NFFT,
    PWELCH_ERROR_INVALID_FS,
    PWELCH_ERROR_INVALID_RANGE,
    PWELCH_ERROR_INVALID_PARENT,
    PWELCH_ERROR_INTERNAL,
];
pub const PWELCH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
