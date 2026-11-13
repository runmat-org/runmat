use crate::*;

pub const FFTN_WIDE_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fftn-wide-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fftn with host int64 or uint64 data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FftnWideIntegerDataExtension"),
};
pub const FFTN_WIDE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fftn-wide-integer-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fftn with int64 or uint64 SIZE controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FftnWideIntegerControlExtension"),
};
pub const FFTN_SHORT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fftn-short-size-vector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fftn SIZE vectors shorter than ndims(X) are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FftnShortSizeExtension"),
};
pub const FFTN_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FFTN_WIDE_DATA_EXTENSION,
    FFTN_WIDE_CONTROL_EXTENSION,
    FFTN_SHORT_SIZE_EXTENSION,
];

pub const FFTN_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::catalog::integer::INTEGER_CLASSES_THROUGH_32_BITS,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Documented integer arrays enter the double N-dimensional FFT domain.",
}];
pub const FFTN_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "SIZE",
    classes: &crate::catalog::integer::INTEGER_CLASSES_THROUGH_32_BITS,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "SIZE is a positive integer vector with length at least ndims(X); logical vectors are also documented.",
}];
pub const FFTN_WIDE_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &[BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Host wide data has its own gate and must be exactly representable as double.",
    }];
pub const FFTN_WIDE_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "SIZE",
        classes: &[BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "Wide SIZE controls have their own gate and are decoded exactly from authoritative storage.",
    },
];
pub const FFTN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fftn(integer_X)",
        inputs: &FFTN_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented integer data produces double output and resident fallback returns to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fftn(X, integer_SIZE)",
        inputs: &FFTN_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "SIZE is decoded exactly and validated for vector shape, positivity, and minimum length before execution.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fftn(int64_or_uint64_X)",
        inputs: &FFTN_WIDE_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only wide host data may not silently round; resident wide data is rejected without an exact provider transform.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fftn(X, int64_or_uint64_SIZE)",
        inputs: &FFTN_WIDE_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only wide SIZE controls are independently gated; short positive vectors have a separate extension gate.",
    },
];

pub const FFTN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "N-D complex Fourier spectrum output.",
}];

pub const FFTN_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

pub const FFTN_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "SIZE",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Positive transform-size vector with length at least ndims(X).",
    },
];

pub const FFTN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = fftn(X)",
        inputs: &FFTN_INPUTS_CORE,
        outputs: &FFTN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fftn(X, SIZE)",
        inputs: &FFTN_INPUTS_SIZE,
        outputs: &FFTN_OUTPUT,
    },
];

pub const FFTN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTN.ARG_COUNT",
    identifier: Some("RunMat:fftn:ArgCount"),
    when: "More than two input arguments are supplied.",
    message: "fftn: invalid argument count",
};

pub const FFTN_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTN.INVALID_SIZE",
    identifier: Some("RunMat:fftn:InvalidSize"),
    when: "SIZE argument is invalid.",
    message: "fftn: invalid SIZE argument",
};

pub const FFTN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTN.INVALID_INPUT",
    identifier: Some("RunMat:fftn:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "fftn: invalid input",
};

pub const FFTN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTN.INTERNAL",
    identifier: Some("RunMat:fftn:Internal"),
    when: "FFTN execution or tensor shaping fails.",
    message: "fftn: internal error",
};

pub const FFTN_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FFTN_ERROR_ARG_COUNT,
    FFTN_ERROR_INVALID_SIZE,
    FFTN_ERROR_INVALID_INPUT,
    FFTN_ERROR_INTERNAL,
];

pub const FFTN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FFTN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FFTN_ERRORS,
};
