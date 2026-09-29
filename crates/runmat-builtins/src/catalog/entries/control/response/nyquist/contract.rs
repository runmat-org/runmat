use crate::*;

const SYS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar SISO tf model.",
};
const W: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "w",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Finite nonnegative angular frequencies.",
};
const INPUT_SYS: [BuiltinParamDescriptor; 1] = [SYS];
const INPUT_SYS_W: [BuiltinParamDescriptor; 2] = [SYS, W];
const RE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "re",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real response column.",
};
const IM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "im",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Imaginary response column.",
};
const WOUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "wout",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Frequency column.",
};
const OUTPUT_RE: [BuiltinParamDescriptor; 1] = [RE];
const OUTPUT_RE_IM: [BuiltinParamDescriptor; 2] = [RE, IM];
const OUTPUT_ALL: [BuiltinParamDescriptor; 3] = [RE, IM, WOUT];
const SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "nyquist(sys)",
        inputs: &INPUT_SYS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "nyquist(sys,w)",
        inputs: &INPUT_SYS_W,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "re = nyquist(sys,w)",
        inputs: &INPUT_SYS_W,
        outputs: &OUTPUT_RE,
    },
    BuiltinSignatureDescriptor {
        label: "[re,im] = nyquist(sys,w)",
        inputs: &INPUT_SYS_W,
        outputs: &OUTPUT_RE_IM,
    },
    BuiltinSignatureDescriptor {
        label: "[re,im,wout] = nyquist(sys,w)",
        inputs: &INPUT_SYS_W,
        outputs: &OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "[re,im,wout] = nyquist(sys)",
        inputs: &INPUT_SYS,
        outputs: &OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "re = nyquist(sys)",
        inputs: &INPUT_SYS,
        outputs: &OUTPUT_RE,
    },
];
pub const NYQUIST_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NYQUIST.INVALID_ARGUMENT",
    identifier: Some("RunMat:nyquist:InvalidArgument"),
    when: "Invocation form, frequency input or output count is invalid.",
    message: "nyquist: invalid argument",
};
pub const NYQUIST_ERROR_INVALID_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NYQUIST.INVALID_MODEL",
    identifier: Some("RunMat:nyquist:InvalidModel"),
    when: "Required model properties are malformed.",
    message: "nyquist: invalid model",
};
pub const NYQUIST_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NYQUIST.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:nyquist:UnsupportedModel"),
    when: "The model is well formed but outside supported SISO tf behavior.",
    message: "nyquist: unsupported model",
};
pub const NYQUIST_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NYQUIST.INTERNAL",
    identifier: Some("RunMat:nyquist:Internal"),
    when: "Numeric response or argument materialization fails internally.",
    message: "nyquist: internal error",
};
pub const NYQUIST_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NYQUIST.PLOT_FAILED",
    identifier: Some("RunMat:nyquist:PlotFailed"),
    when: "Statement-form plotting fails for reasons other than known nonfatal setup conditions.",
    message: "nyquist: plotting failed",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    NYQUIST_ERROR_INVALID_ARGUMENT,
    NYQUIST_ERROR_INVALID_MODEL,
    NYQUIST_ERROR_UNSUPPORTED_MODEL,
    NYQUIST_ERROR_INTERNAL,
    NYQUIST_ERROR_PLOT_FAILED,
];
pub const NYQUIST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const NYQUIST_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "nyquist-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Native typed-integer frequency or model metadata is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NyquistIntegerExtension"),
};
pub const NYQUIST_LOGICAL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "nyquist-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Logical frequency or model metadata is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NyquistLogicalExtension"),
};
pub const NYQUIST_RESIDENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "nyquist-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "GPU-resident frequency or model metadata is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NyquistResidentExtension"),
};
pub const NYQUIST_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    NYQUIST_INTEGER_EXTENSION,
    NYQUIST_LOGICAL_EXTENSION,
    NYQUIST_RESIDENT_EXTENSION,
];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "sys numeric metadata", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Native integer coefficients and timing metadata must convert exactly before complex binary64 evaluation." },
    BuiltinIntegerInputCapability { name: "w", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Native integer frequencies must convert exactly and remain nonnegative." },
];
pub const NYQUIST_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[re,im,wout] = nyquist(sys, integer_w?)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes:
            "Mode admission and exact binary64 conversion occur before host frequency evaluation.",
    }];
