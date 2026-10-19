use crate::*;

const PERIODOGRAM_OUTPUT_PXX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pxx",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Power spectral density or power spectrum estimate.",
}];

const PERIODOGRAM_OUTPUT_PXX_F: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "pxx",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Power spectral density or power spectrum estimate.",
    },
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Frequency vector in radians/sample or Hz when fs is supplied.",
    },
];

const PERIODOGRAM_INPUTS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal vector or matrix with one signal per column. In RunMat its floating precision determines every numerical output class.",
    },
    BuiltinParamDescriptor {
        name: "window",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Window vector. [] uses a rectangular window matching the signal length.",
    },
    BuiltinParamDescriptor {
        name: "nfft",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "DFT length or explicit frequency vector.",
    },
    BuiltinParamDescriptor {
        name: "fs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Sampling frequency in Hz.",
    },
    BuiltinParamDescriptor {
        name: "freqrange",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Frequency range selector (`onesided`, `twosided`, or `centered`) and spectrum type (`psd` or `power`).",
    },
];

const PERIODOGRAM_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "pxx = periodogram(x, window, nfft, fs, freqrange)",
        inputs: &PERIODOGRAM_INPUTS,
        outputs: &PERIODOGRAM_OUTPUT_PXX,
    },
    BuiltinSignatureDescriptor {
        label: "[pxx, f] = periodogram(x, window, nfft, fs, freqrange)",
        inputs: &PERIODOGRAM_INPUTS,
        outputs: &PERIODOGRAM_OUTPUT_PXX_F,
    },
];

pub const PERIODOGRAM_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.ARG_COUNT",
    identifier: Some("RunMat:periodogram:ArgCount"),
    when: "The argument count is outside supported forms.",
    message:
        "periodogram: expected periodogram(x, [window, [nfft_or_f, [fs, [freqrange_or_scale]]]])",
};

pub const PERIODOGRAM_ERROR_INVALID_SIGNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.INVALID_SIGNAL",
    identifier: Some("RunMat:periodogram:InvalidSignal"),
    when: "Input signal is not a numeric vector or matrix.",
    message: "periodogram: x must be a nonempty numeric vector or 2-D matrix",
};

pub const PERIODOGRAM_ERROR_INVALID_WINDOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.INVALID_WINDOW",
    identifier: Some("RunMat:periodogram:InvalidWindow"),
    when: "Window input is invalid.",
    message: "periodogram: window must be a finite real vector matching the signal length",
};

pub const PERIODOGRAM_ERROR_INVALID_NFFT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.INVALID_NFFT",
    identifier: Some("RunMat:periodogram:InvalidNfft"),
    when: "DFT length or frequency vector is invalid.",
    message: "periodogram: nfft must be a positive integer or finite real frequency vector",
};

pub const PERIODOGRAM_ERROR_INVALID_FS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.INVALID_FS",
    identifier: Some("RunMat:periodogram:InvalidFs"),
    when: "Sampling frequency is invalid.",
    message: "periodogram: fs must be a positive finite scalar",
};

pub const PERIODOGRAM_ERROR_INVALID_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.INVALID_RANGE",
    identifier: Some("RunMat:periodogram:InvalidRange"),
    when: "Frequency range selector or spectrum type is invalid.",
    message: "periodogram: freqrange must be 'onesided', 'twosided', or 'centered'; spectrum type must be 'psd' or 'power'",
};

pub const PERIODOGRAM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.INTERNAL",
    identifier: Some("RunMat:periodogram:Internal"),
    when: "Output tensor construction fails internally.",
    message: "periodogram: internal error",
};

const PERIODOGRAM_ERRORS: [BuiltinErrorDescriptor; 7] = [
    PERIODOGRAM_ERROR_ARG_COUNT,
    PERIODOGRAM_ERROR_INVALID_SIGNAL,
    PERIODOGRAM_ERROR_INVALID_WINDOW,
    PERIODOGRAM_ERROR_INVALID_NFFT,
    PERIODOGRAM_ERROR_INVALID_FS,
    PERIODOGRAM_ERROR_INVALID_RANGE,
    PERIODOGRAM_ERROR_INTERNAL,
];

pub const PERIODOGRAM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PERIODOGRAM_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PERIODOGRAM_ERRORS,
};
