use crate::*;
const FREQZ_OUTPUT_H: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "H",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex frequency response.",
}];

const FREQZ_OUTPUT_H_W: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "H",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex frequency response.",
    },
    BuiltinParamDescriptor {
        name: "w",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Frequencies in radians/sample or Hz when fs is supplied.",
    },
];

const FREQZ_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
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
];

const FREQZ_INPUTS_N: [BuiltinParamDescriptor; 3] = [
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
        name: "n",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("512"),
        description: "Number of response samples.",
    },
];

const FREQZ_INPUTS_N_FS: [BuiltinParamDescriptor; 4] = [
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
        name: "n",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("512"),
        description: "Number of response samples.",
    },
    BuiltinParamDescriptor {
        name: "fs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Sampling frequency for output frequencies in Hz.",
    },
];

const FREQZ_OUTPUT_NONE: [BuiltinParamDescriptor; 0] = [];

const FREQZ_SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "freqz(b, a)",
        inputs: &FREQZ_INPUTS_CORE,
        outputs: &FREQZ_OUTPUT_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "freqz(b, a, n)",
        inputs: &FREQZ_INPUTS_N,
        outputs: &FREQZ_OUTPUT_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "freqz(b, a, n, fs)",
        inputs: &FREQZ_INPUTS_N_FS,
        outputs: &FREQZ_OUTPUT_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "H = freqz(b, a)",
        inputs: &FREQZ_INPUTS_CORE,
        outputs: &FREQZ_OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "H = freqz(b, a, n)",
        inputs: &FREQZ_INPUTS_N,
        outputs: &FREQZ_OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "H = freqz(b, a, n, fs)",
        inputs: &FREQZ_INPUTS_N_FS,
        outputs: &FREQZ_OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "[H, w] = freqz(b, a)",
        inputs: &FREQZ_INPUTS_CORE,
        outputs: &FREQZ_OUTPUT_H_W,
    },
    BuiltinSignatureDescriptor {
        label: "[H, w] = freqz(b, a, n)",
        inputs: &FREQZ_INPUTS_N,
        outputs: &FREQZ_OUTPUT_H_W,
    },
    BuiltinSignatureDescriptor {
        label: "[H, w] = freqz(b, a, n, fs)",
        inputs: &FREQZ_INPUTS_N_FS,
        outputs: &FREQZ_OUTPUT_H_W,
    },
];

pub const FREQZ_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.ARG_COUNT",
    identifier: Some("RunMat:freqz:ArgCount"),
    when: "The argument count is outside supported forms.",
    message: "freqz: expected freqz(b, a, [n, [fs]])",
};

pub const FREQZ_ERROR_INVALID_COEFFICIENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.INVALID_COEFFICIENTS",
    identifier: Some("RunMat:freqz:InvalidCoefficients"),
    when: "Coefficient inputs are empty or not numeric vectors.",
    message: "freqz: invalid coefficient input",
};

pub const FREQZ_ERROR_INVALID_N: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.INVALID_N",
    identifier: Some("RunMat:freqz:InvalidN"),
    when: "The response length is not an integer scalar greater than or equal to two.",
    message: "freqz: n must be an integer greater than or equal to 2",
};

pub const FREQZ_ERROR_INVALID_FS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.INVALID_FS",
    identifier: Some("RunMat:freqz:InvalidFs"),
    when: "The sampling frequency is not a positive finite scalar.",
    message: "freqz: fs must be a positive finite scalar",
};

pub const FREQZ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.INTERNAL",
    identifier: Some("RunMat:freqz:Internal"),
    when: "Response tensor construction fails internally.",
    message: "freqz: internal error",
};

pub const FREQZ_ERROR_PLOT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.PLOT",
    identifier: Some("RunMat:freqz:Plot"),
    when: "A zero-output response cannot be prepared or presented.",
    message: "freqz: could not present frequency response",
};

const FREQZ_ERRORS: [BuiltinErrorDescriptor; 6] = [
    FREQZ_ERROR_ARG_COUNT,
    FREQZ_ERROR_INVALID_COEFFICIENTS,
    FREQZ_ERROR_INVALID_N,
    FREQZ_ERROR_INVALID_FS,
    FREQZ_ERROR_INTERNAL,
    FREQZ_ERROR_PLOT,
];

macro_rules! freqz_extension {
    ($name:ident, $id:literal, $description:literal, $error:literal) => {
        pub const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $description,
            error_identifier: Some($error),
        };
    };
}

freqz_extension!(
    FREQZ_INTEGER_COEFFICIENT_EXTENSION,
    "freqz-integer-coefficients",
    "freqz with typed-integer transfer-function coefficients is a RunMat extension",
    "RunMat:compatibility:FreqzIntegerCoefficientExtension"
);
freqz_extension!(
    FREQZ_LOGICAL_COEFFICIENT_EXTENSION,
    "freqz-logical-coefficients",
    "freqz with logical transfer-function coefficients is a RunMat extension",
    "RunMat:compatibility:FreqzLogicalCoefficientExtension"
);
freqz_extension!(
    FREQZ_INTEGER_N_EXTENSION,
    "freqz-integer-point-count",
    "freqz with a typed-integer point count is a provisional RunMat extension because the public core documentation does not publish its accepted classes",
    "RunMat:compatibility:FreqzIntegerPointCountExtension"
);
freqz_extension!(
    FREQZ_LOGICAL_N_EXTENSION,
    "freqz-logical-point-count",
    "freqz with a logical point count is a RunMat extension",
    "RunMat:compatibility:FreqzLogicalPointCountExtension"
);
freqz_extension!(
    FREQZ_SINGLE_N_EXTENSION,
    "freqz-single-point-count",
    "freqz with a single-precision point count is a provisional RunMat extension",
    "RunMat:compatibility:FreqzSinglePointCountExtension"
);
freqz_extension!(
    FREQZ_INTEGER_FS_EXTENSION,
    "freqz-integer-sample-rate",
    "freqz with a typed-integer sample rate is a RunMat extension",
    "RunMat:compatibility:FreqzIntegerSampleRateExtension"
);
freqz_extension!(
    FREQZ_LOGICAL_FS_EXTENSION,
    "freqz-logical-sample-rate",
    "freqz with a logical sample rate is a RunMat extension",
    "RunMat:compatibility:FreqzLogicalSampleRateExtension"
);
freqz_extension!(
    FREQZ_SINGLE_FS_EXTENSION,
    "freqz-single-sample-rate",
    "freqz with a single-precision sample rate is a RunMat extension",
    "RunMat:compatibility:FreqzSingleSampleRateExtension"
);
freqz_extension!(
    FREQZ_RESIDENT_INPUT_EXTENSION,
    "freqz-resident-input",
    "freqz with an interactive resident input is a RunMat extension",
    "RunMat:compatibility:FreqzResidentInputExtension"
);

pub const FREQZ_EXTENSIONS: [BuiltinExtensionDescriptor; 9] = [
    FREQZ_INTEGER_COEFFICIENT_EXTENSION,
    FREQZ_LOGICAL_COEFFICIENT_EXTENSION,
    FREQZ_INTEGER_N_EXTENSION,
    FREQZ_LOGICAL_N_EXTENSION,
    FREQZ_SINGLE_N_EXTENSION,
    FREQZ_INTEGER_FS_EXTENSION,
    FREQZ_LOGICAL_FS_EXTENSION,
    FREQZ_SINGLE_FS_EXTENSION,
    FREQZ_RESIDENT_INPUT_EXTENSION,
];

const FREQZ_INTEGER_COEFFICIENT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "b",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer numerator coefficients must be exactly representable at the binary64 response boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "a",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer denominator coefficients must be exactly representable at the binary64 response boundary.",
    },
];
const FREQZ_INTEGER_N_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "n",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Core freqz documents a positive integer scalar but publishes no datatype list; RunMat provisionally gates typed n and parses it exactly.",
}];
const FREQZ_INTEGER_FS_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "fs",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented sample-rate class is double; admitted typed values require exact binary64 representation.",
    },
];

pub const FREQZ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[h,w] = freqz(integer_b,integer_a,...)",
        inputs: &FREQZ_INTEGER_COEFFICIENT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Authoritative integer coefficients cross one checked binary64 boundary; integer-only responses are complex double and frequencies are double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[h,w] = freqz(b,a,integer_n,...)",
        inputs: &FREQZ_INTEGER_N_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Point count is decoded exactly in every supported integer class, must be at least two, and selects the response shape.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[h,f] = freqz(b,a,n,integer_fs)",
        inputs: &FREQZ_INTEGER_FS_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Sample rate crosses one checked binary64 boundary; response precision follows documented coefficient precision and f remains host double.",
    },
];

pub const FREQZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FREQZ_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FREQZ_ERRORS,
};
