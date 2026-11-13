use crate::*;

pub const FFT_WIDE_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fft-wide-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fft with host int64 or uint64 data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FftWideIntegerDataExtension"),
};

pub const FFT_WIDE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fft-wide-integer-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fft with int64 or uint64 N or DIM controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FftWideIntegerControlExtension"),
};

pub const FFT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [FFT_WIDE_DATA_EXTENSION, FFT_WIDE_CONTROL_EXTENSION];

pub const FFT_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::catalog::integer::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented integer data enters the floating FFT domain and produces double output.",
    }];

pub const FFT_WIDE_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &[BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Host wide integers are independently gated and must be exactly representable as double before FFT evaluation; resident wide integers are rejected.",
    },
];

pub const FFT_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "N",
        classes: &crate::catalog::integer::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "N is parsed exactly as a nonnegative structural length; logical scalar controls are also documented.",
    },
    BuiltinIntegerInputCapability {
        name: "DIM",
        classes: &crate::catalog::integer::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "DIM is parsed exactly as a positive one-based dimension; logical scalar controls are also documented.",
    },
];

pub const FFT_WIDE_INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "N or DIM",
        classes: &[BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "Wide structural controls are decoded from authoritative integer storage and independently gated.",
    },
];

pub const FFT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fft(integer_X, ...)",
        inputs: &FFT_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer input is converted once at the documented double FFT boundary; provider fallback restores a resident result to the input owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fft(X, integer_N, integer_DIM)",
        inputs: &FFT_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Documented controls are decoded exactly in every supported integer class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fft(int64_or_uint64_X, ...)",
        inputs: &FFT_WIDE_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only wide data cannot silently round at the double boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fft(X, int64_or_uint64_N_or_DIM)",
        inputs: &FFT_WIDE_INTEGER_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only wide controls are independently gated and parsed exactly.",
    },
];

pub const FFT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex Fourier spectrum output.",
}];

pub const FFT_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input signal/array.",
}];

pub const FFT_INPUTS_WITH_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal/array.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Transform length along selected dimension.",
    },
];

pub const FFT_INPUTS_WITH_N_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal/array.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Transform length along selected dimension.",
    },
    BuiltinParamDescriptor {
        name: "DIM",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("first non-singleton dimension"),
        description: "Dimension to transform along.",
    },
];

pub const FFT_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = fft(X)",
        inputs: &FFT_INPUTS_CORE,
        outputs: &FFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fft(X, N)",
        inputs: &FFT_INPUTS_WITH_N,
        outputs: &FFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fft(X, N, DIM)",
        inputs: &FFT_INPUTS_WITH_N_DIM,
        outputs: &FFT_OUTPUT,
    },
];

pub const FFT_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT.ARG_COUNT",
    identifier: Some("RunMat:fft:ArgCount"),
    when: "More than three input arguments are supplied.",
    message: "fft: expected fft(X), fft(X, N), or fft(X, N, DIM)",
};

pub const FFT_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT.INVALID_LENGTH",
    identifier: Some("RunMat:fft:InvalidLength"),
    when: "Length argument N is invalid.",
    message: "fft: invalid length argument",
};

pub const FFT_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT.INVALID_DIMENSION",
    identifier: Some("RunMat:fft:InvalidDimension"),
    when: "Dimension argument DIM is invalid.",
    message: "fft: invalid dimension argument",
};

pub const FFT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT.INVALID_INPUT",
    identifier: Some("RunMat:fft:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "fft: invalid input",
};

pub const FFT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT.INTERNAL",
    identifier: Some("RunMat:fft:Internal"),
    when: "FFT execution or tensor shaping fails.",
    message: "fft: internal error",
};

pub const FFT_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FFT_ERROR_ARG_COUNT,
    FFT_ERROR_INVALID_LENGTH,
    FFT_ERROR_INVALID_DIMENSION,
    FFT_ERROR_INVALID_INPUT,
    FFT_ERROR_INTERNAL,
];

pub const FFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FFT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FFT_ERRORS,
};
