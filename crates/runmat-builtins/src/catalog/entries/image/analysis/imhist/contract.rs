use crate::*;

/// Runtime and catalog share the same supported histogram count policy.
pub const IMHIST_MAX_BINS: usize = 1_000_000;
pub const IMHIST_INTEGER_TOLERANCE: f64 = 1.0e-9;

const IMHIST_OUTPUT_COUNTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "counts",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Histogram bin counts as a column vector.",
}];

const IMHIST_OUTPUT_COUNTS_BINS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "counts",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Histogram bin counts as a column vector.",
    },
    BuiltinParamDescriptor {
        name: "binLocations",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Intensity or colormap-index bin locations as a column vector.",
    },
];

const IMHIST_INPUTS_IMAGE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grayscale intensity image.",
}];

const IMHIST_INPUTS_IMAGE_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grayscale intensity image.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: Some("256"),
        description: "Number of bins.",
    },
];

const IMHIST_INPUTS_INDEXED: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexed image matrix.",
    },
    BuiltinParamDescriptor {
        name: "map",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Colormap with one RGB row per indexed-image bin.",
    },
];

const IMHIST_SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "imhist(I)",
        inputs: &IMHIST_INPUTS_IMAGE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "imhist(I, n)",
        inputs: &IMHIST_INPUTS_IMAGE_N,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "imhist(X, map)",
        inputs: &IMHIST_INPUTS_INDEXED,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "counts = imhist(I)",
        inputs: &IMHIST_INPUTS_IMAGE,
        outputs: &IMHIST_OUTPUT_COUNTS,
    },
    BuiltinSignatureDescriptor {
        label: "counts = imhist(I, n)",
        inputs: &IMHIST_INPUTS_IMAGE_N,
        outputs: &IMHIST_OUTPUT_COUNTS,
    },
    BuiltinSignatureDescriptor {
        label: "[counts, binLocations] = imhist(I)",
        inputs: &IMHIST_INPUTS_IMAGE,
        outputs: &IMHIST_OUTPUT_COUNTS_BINS,
    },
    BuiltinSignatureDescriptor {
        label: "[counts, binLocations] = imhist(I, n)",
        inputs: &IMHIST_INPUTS_IMAGE_N,
        outputs: &IMHIST_OUTPUT_COUNTS_BINS,
    },
    BuiltinSignatureDescriptor {
        label: "counts = imhist(X, map)",
        inputs: &IMHIST_INPUTS_INDEXED,
        outputs: &IMHIST_OUTPUT_COUNTS,
    },
    BuiltinSignatureDescriptor {
        label: "[counts, binLocations] = imhist(X, map)",
        inputs: &IMHIST_INPUTS_INDEXED,
        outputs: &IMHIST_OUTPUT_COUNTS_BINS,
    },
];

pub const IMHIST_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMHIST.INVALID_ARGUMENT",
    identifier: Some("RunMat:imhist:InvalidArgument"),
    when: "Image input, bin count, or colormap arguments are malformed or unsupported.",
    message: "imhist: invalid argument",
};

pub const IMHIST_ERROR_UNSUPPORTED_IMAGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMHIST.UNSUPPORTED_IMAGE",
    identifier: Some("RunMat:imhist:UnsupportedImage"),
    when: "Input cannot be interpreted as a grayscale or indexed image.",
    message: "imhist: unsupported image input",
};

pub const IMHIST_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMHIST.PLOT_FAILED",
    identifier: Some("RunMat:imhist:PlotFailed"),
    when: "Statement-form histogram rendering fails.",
    message: "imhist: plotting failed",
};

pub const IMHIST_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMHIST.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:imhist:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "imhist: too many output arguments",
};

pub const IMHIST_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMHIST.INTERNAL",
    identifier: Some("RunMat:imhist:Internal"),
    when: "Internal histogram assembly fails.",
    message: "imhist: internal error",
};

const IMHIST_ERRORS: [BuiltinErrorDescriptor; 5] = [
    IMHIST_ERROR_INVALID_ARGUMENT,
    IMHIST_ERROR_UNSUPPORTED_IMAGE,
    IMHIST_ERROR_PLOT_FAILED,
    IMHIST_ERROR_TOO_MANY_OUTPUTS,
    IMHIST_ERROR_INTERNAL,
];

pub const IMHIST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMHIST_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMHIST_ERRORS,
};

pub const IMHIST_TYPED_BIN_COUNT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "imhist-typed-integer-bin-count",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "imhist accepts a typed-integer bin count as a RunMat extension because the public positive-integer form does not enumerate typed storage classes",
    error_identifier: Some("RunMat:compatibility:ImhistTypedIntegerBinCountExtension"),
};
pub const IMHIST_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [IMHIST_TYPED_BIN_COUNT_EXTENSION];

const IMHIST_GRAYSCALE_INTEGER_CLASSES: [BuiltinIntegerClass; 6] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
    BuiltinIntegerClass::Uint32,
];
const IMHIST_WIDE_INTEGER_CLASSES: [BuiltinIntegerClass; 2] =
    [BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64];
const IMHIST_INDEXED_INTEGER_CLASSES: [BuiltinIntegerClass; 2] =
    [BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16];
const IMHIST_GRAYSCALE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &IMHIST_GRAYSCALE_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Documented grayscale integer image classes use their full class interval. RunMat currently returns double counts and bin locations; the exact public output classes remain evidence-open below.",
    }];
const IMHIST_INDEXED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &IMHIST_INDEXED_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Uint8 and uint16 indexed images use zero-based colormap indices; indexed gpuArray input is not supported.",
    }];
const IMHIST_BIN_COUNT_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public form requires a positive integer scalar but does not enumerate typed-integer storage classes; RunMat accepts every native integer scalar exactly after range validation.",
    }];
const IMHIST_REJECTED_WIDE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I_or_X",
        classes: &IMHIST_WIDE_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Int64 and uint64 are outside both documented image-class surfaces and reject from host or resident metadata before evaluation.",
    }];
pub const IMHIST_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "[counts, binLocations] = imhist(integer_I, n?)", inputs: &IMHIST_GRAYSCALE_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Exact native values are assigned to the documented class-wide half-open bins. Counts and bin locations are double, and supported resident grayscale results use owner-aware fallback." },
    BuiltinIntegerCapabilityDescriptor { form: "[counts, binLocations] = imhist(integer_X, map)", inputs: &IMHIST_INDEXED_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Host uint8 and uint16 indices are counted exactly, counts and bin locations are double, and indexed gpuArray input is rejected as unsupported." },
    BuiltinIntegerCapabilityDescriptor { form: "[counts, binLocations] = imhist(I, integer_n)", inputs: &IMHIST_BIN_COUNT_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "The public reference requires a positive integer-valued bin count without enumerating native storage classes. RunMat accepts all eight native integer classes only as a gated extension, parses them exactly, and returns double histogram arrays." },
    BuiltinIntegerCapabilityDescriptor { form: "imhist(int64_or_uint64_image, ...)", inputs: &IMHIST_REJECTED_WIDE_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Unsupported wide image classes reject without a lossy floating conversion or provider access." },
];
