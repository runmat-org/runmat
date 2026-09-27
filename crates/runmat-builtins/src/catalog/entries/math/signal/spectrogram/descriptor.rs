use crate::*;

const SPECTROGRAM_OUTPUT_S: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex short-time Fourier transform matrix in the signal's floating precision, like every numerical output in RunMat.",
}];

const SPECTROGRAM_OUTPUT_FULL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "s",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex short-time Fourier transform matrix.",
    },
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Frequencies in radians/sample or Hz when fs is supplied.",
    },
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Segment midpoint sample indices or time instants when fs is supplied.",
    },
    BuiltinParamDescriptor {
        name: "ps",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Power spectral density or power spectrum estimate for each segment.",
    },
];

const SPECTROGRAM_INPUTS: [BuiltinParamDescriptor; 7] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal vector.",
    },
    BuiltinParamDescriptor {
        name: "win",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Hamming-window length, window vector, or [] for the default window.",
    },
    BuiltinParamDescriptor {
        name: "nOverlap",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Number of overlapped samples between adjacent segments.",
    },
    BuiltinParamDescriptor {
        name: "freqSpec",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "DFT length or explicit frequency vector.",
    },
    BuiltinParamDescriptor {
        name: "Fs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Sampling frequency in Hz.",
    },
    BuiltinParamDescriptor {
        name: "freqRange",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Frequency range selector: `onesided`, `twosided`, or `centered`.",
    },
    BuiltinParamDescriptor {
        name: "spectrumType",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("psd"),
        description: "Spectral estimate scaling: `psd` or `power`.",
    },
];

const SPECTROGRAM_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "s = spectrogram(x, win, nOverlap, freqSpec, Fs, freqRange, spectrumType)",
        inputs: &SPECTROGRAM_INPUTS,
        outputs: &SPECTROGRAM_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label:
            "[s, f, t, ps] = spectrogram(x, win, nOverlap, freqSpec, Fs, freqRange, spectrumType)",
        inputs: &SPECTROGRAM_INPUTS,
        outputs: &SPECTROGRAM_OUTPUT_FULL,
    },
];

pub const SPECTROGRAM_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.ARG_COUNT",
    identifier: Some("RunMat:spectrogram:ArgCount"),
    when: "The argument count is outside supported forms.",
    message: "spectrogram: expected spectrogram(x, [win, [nOverlap, [freqSpec, [Fs, [freqRange_or_spectrumType]]]]])",
};

pub const SPECTROGRAM_ERROR_INVALID_SIGNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INVALID_SIGNAL",
    identifier: Some("RunMat:spectrogram:InvalidSignal"),
    when: "Input signal is not a numeric vector.",
    message: "spectrogram: x must be a nonempty numeric vector",
};

pub const SPECTROGRAM_ERROR_INVALID_WINDOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INVALID_WINDOW",
    identifier: Some("RunMat:spectrogram:InvalidWindow"),
    when: "Window input is invalid.",
    message:
        "spectrogram: win must be [] or a nonempty finite real vector or positive integer length",
};

pub const SPECTROGRAM_ERROR_INVALID_OVERLAP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INVALID_OVERLAP",
    identifier: Some("RunMat:spectrogram:InvalidOverlap"),
    when: "Overlap is invalid.",
    message: "spectrogram: nOverlap must be a nonnegative integer smaller than the window length",
};

pub const SPECTROGRAM_ERROR_INVALID_FREQ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INVALID_FREQ",
    identifier: Some("RunMat:spectrogram:InvalidFreqSpec"),
    when: "Frequency specification is invalid.",
    message: "spectrogram: freqSpec must be a positive integer or finite real frequency vector with at least two elements",
};

pub const SPECTROGRAM_ERROR_INVALID_FS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INVALID_FS",
    identifier: Some("RunMat:spectrogram:InvalidFs"),
    when: "Sampling frequency is invalid.",
    message: "spectrogram: Fs must be a positive finite scalar",
};

pub const SPECTROGRAM_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INVALID_OPTION",
    identifier: Some("RunMat:spectrogram:InvalidOption"),
    when: "Frequency range, spectrum type, or display option is invalid.",
    message: "spectrogram: expected 'onesided', 'twosided', 'centered', 'psd', 'power', 'reassigned', or 'yaxis'",
};

pub const SPECTROGRAM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INTERNAL",
    identifier: Some("RunMat:spectrogram:Internal"),
    when: "Output tensor construction fails internally.",
    message: "spectrogram: internal error",
};

const SPECTROGRAM_ERRORS: [BuiltinErrorDescriptor; 8] = [
    SPECTROGRAM_ERROR_ARG_COUNT,
    SPECTROGRAM_ERROR_INVALID_SIGNAL,
    SPECTROGRAM_ERROR_INVALID_WINDOW,
    SPECTROGRAM_ERROR_INVALID_OVERLAP,
    SPECTROGRAM_ERROR_INVALID_FREQ,
    SPECTROGRAM_ERROR_INVALID_FS,
    SPECTROGRAM_ERROR_INVALID_OPTION,
    SPECTROGRAM_ERROR_INTERNAL,
];

pub const SPECTROGRAM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPECTROGRAM_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPECTROGRAM_ERRORS,
};
