use crate::*;

pub const FFT2_WIDE_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fft2-wide-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fft2 with host int64 or uint64 data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Fft2WideIntegerDataExtension"),
};
pub const FFT2_WIDE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fft2-wide-integer-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fft2 with int64 or uint64 transform-size controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Fft2WideIntegerControlExtension"),
};
pub const FFT2_SIZE_FORM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fft2-size-form",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fft2(X, SIZE) scalar/vector shorthand is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Fft2SizeFormExtension"),
};
pub const FFT2_EMPTY_ZERO_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fft2-empty-zero-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fft2 empty or zero transform sizes are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Fft2EmptyZeroSizeExtension"),
};
pub const FFT2_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FFT2_WIDE_DATA_EXTENSION,
    FFT2_WIDE_CONTROL_EXTENSION,
    FFT2_SIZE_FORM_EXTENSION,
    FFT2_EMPTY_ZERO_SIZE_EXTENSION,
];

pub const FFT2_DOCUMENTED_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::catalog::integer::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented integer arrays enter the double two-dimensional FFT domain.",
    }];
pub const FFT2_DOCUMENTED_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "M and N",
        classes: &crate::catalog::integer::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "M and N are exact positive scalar structural controls; logical scalars are also documented.",
    },
];
pub const FFT2_WIDE_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &[BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Host wide data has its own gate and must cross the double boundary exactly.",
    }];
pub const FFT2_WIDE_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "M, N, or SIZE",
        classes: &[BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "Wide structural controls have their own gate and are parsed exactly from authoritative storage.",
    },
];
pub const FFT2_SIZE_EXTENSION_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "SIZE",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "The scalar or two-element SIZE shorthand is independently mode-gated and parsed from authoritative storage.",
    },
];
pub const FFT2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fft2(integer_X)",
        inputs: &FFT2_DOCUMENTED_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented integer data produces double output; GPU fallback restores residency to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fft2(X, integer_M, integer_N)",
        inputs: &FFT2_DOCUMENTED_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Documented positive controls are parsed exactly before allocation or provider execution.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fft2(int64_or_uint64_X, ...)",
        inputs: &FFT2_WIDE_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only wide host data may not silently round; resident wide data is rejected without an exact provider transform.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fft2(X, int64_or_uint64_M_or_N_or_SIZE)",
        inputs: &FFT2_WIDE_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only wide controls are independently gated and decoded exactly before execution.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fft2(X, integer_SIZE)",
        inputs: &FFT2_SIZE_EXTENSION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only scalar/two-vector shorthand; empty and zero sizes have a further independent extension gate.",
    },
];

pub const FFT2_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "2-D complex Fourier spectrum output.",
}];

pub const FFT2_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

pub const FFT2_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [
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
        description: "RunMat-only scalar N or two-element [M N] size shorthand.",
    },
];

pub const FFT2_INPUTS_M_N: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Positive output row count for transform.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Positive output column count for transform.",
    },
];

pub const FFT2_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = fft2(X)",
        inputs: &FFT2_INPUTS_CORE,
        outputs: &FFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fft2(X, SIZE)",
        inputs: &FFT2_INPUTS_SIZE,
        outputs: &FFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fft2(X, M, N)",
        inputs: &FFT2_INPUTS_M_N,
        outputs: &FFT2_OUTPUT,
    },
];

pub const FFT2_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.ARG_COUNT",
    identifier: Some("RunMat:fft2:ArgCount"),
    when: "More than three input arguments are supplied.",
    message: "fft2: invalid argument count",
};

pub const FFT2_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.INVALID_LENGTH",
    identifier: Some("RunMat:fft2:InvalidLength"),
    when: "Length/size arguments are invalid.",
    message: "fft2: invalid transform length argument",
};

pub const FFT2_ERROR_INVALID_SIZE_VECTOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.INVALID_SIZE_VECTOR",
    identifier: Some("RunMat:fft2:InvalidSizeVector"),
    when: "Single SIZE argument is invalid.",
    message: "fft2: invalid size vector argument",
};

pub const FFT2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.INVALID_INPUT",
    identifier: Some("RunMat:fft2:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "fft2: invalid input",
};

pub const FFT2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.INTERNAL",
    identifier: Some("RunMat:fft2:Internal"),
    when: "FFT2 execution or tensor shaping fails.",
    message: "fft2: internal error",
};

pub const FFT2_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FFT2_ERROR_ARG_COUNT,
    FFT2_ERROR_INVALID_LENGTH,
    FFT2_ERROR_INVALID_SIZE_VECTOR,
    FFT2_ERROR_INVALID_INPUT,
    FFT2_ERROR_INTERNAL,
];

pub const FFT2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FFT2_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FFT2_ERRORS,
};
