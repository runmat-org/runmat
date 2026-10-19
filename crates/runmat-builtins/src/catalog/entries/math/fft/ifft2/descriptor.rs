use crate::*;

pub const IFFT2_WIDE_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifft2-wide-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifft2 with host int64 or uint64 data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Ifft2WideIntegerDataExtension"),
};
pub const IFFT2_WIDE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifft2-wide-integer-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifft2 with int64 or uint64 transform-size controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Ifft2WideIntegerControlExtension"),
};
pub const IFFT2_SIZE_FORM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifft2-size-form",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifft2(X, SIZE) scalar/vector shorthand is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Ifft2SizeFormExtension"),
};
pub const IFFT2_EMPTY_ZERO_SIZE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ifft2-empty-zero-size",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ifft2 empty or zero transform sizes are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Ifft2EmptyZeroSizeExtension"),
    };
pub const IFFT2_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    IFFT2_WIDE_DATA_EXTENSION,
    IFFT2_WIDE_CONTROL_EXTENSION,
    IFFT2_SIZE_FORM_EXTENSION,
    IFFT2_EMPTY_ZERO_SIZE_EXTENSION,
];

pub const IFFT2_DOCUMENTED_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::catalog::integer::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented integer arrays enter the double two-dimensional inverse-FFT domain.",
    }];
pub const IFFT2_DOCUMENTED_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "M and N",
        classes: &crate::catalog::integer::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "M and N are exact positive scalar structural controls; logical scalars are also documented.",
    },
];
pub const IFFT2_WIDE_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &[BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Host wide data is gated and must cross the double boundary exactly.",
    }];
pub const IFFT2_WIDE_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "M, N, or SIZE",
        classes: &[BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "Wide structural controls are independently gated and parsed from authoritative integer storage.",
    },
];
pub const IFFT2_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "SIZE",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
    notes: "The scalar or two-element SIZE shorthand is independently gated.",
}];
pub const IFFT2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifft2(integer_X, ...)",
        inputs: &IFFT2_DOCUMENTED_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented integer data produces double output; symmetric changes complexity only.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifft2(X, integer_M, integer_N, ...)",
        inputs: &IFFT2_DOCUMENTED_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Documented controls are parsed exactly before provider execution.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifft2(int64_or_uint64_X, ...)",
        inputs: &IFFT2_WIDE_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only wide host data may not silently round; resident wide data rejects before gather.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifft2(X, int64_or_uint64_M_or_N_or_SIZE, ...)",
        inputs: &IFFT2_WIDE_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only wide controls are independently gated and decoded exactly.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifft2(X, integer_SIZE, ...)",
        inputs: &IFFT2_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only scalar/two-vector shorthand; empty and zero sizes have a further gate.",
    },
];

pub const IFFT2_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "2-D inverse FFT output.",
}];

pub const IFFT2_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input spectrum or signal.",
}];

pub const IFFT2_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "SIZE",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Scalar N or two-element [M N] size vector.",
    },
];

pub const IFFT2_INPUTS_M_N: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output row count for transform.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output column count for transform.",
    },
];

pub const IFFT2_INPUTS_SYMFLAG: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

pub const IFFT2_INPUTS_SIZE_SYMFLAG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "SIZE",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Scalar N or two-element [M N] size vector.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

pub const IFFT2_INPUTS_M_N_SYMFLAG: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output row count for transform.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output column count for transform.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

pub const IFFT2_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X)",
        inputs: &IFFT2_INPUTS_CORE,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, SIZE)",
        inputs: &IFFT2_INPUTS_SIZE,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, M, N)",
        inputs: &IFFT2_INPUTS_M_N,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, symflag)",
        inputs: &IFFT2_INPUTS_SYMFLAG,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, SIZE, symflag)",
        inputs: &IFFT2_INPUTS_SIZE_SYMFLAG,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, M, N, symflag)",
        inputs: &IFFT2_INPUTS_M_N_SYMFLAG,
        outputs: &IFFT2_OUTPUT,
    },
];

pub const IFFT2_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.ARG_COUNT",
    identifier: Some("RunMat:ifft2:ArgCount"),
    when: "More than four input arguments are supplied.",
    message: "ifft2: invalid argument count",
};

pub const IFFT2_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INVALID_LENGTH",
    identifier: Some("RunMat:ifft2:InvalidLength"),
    when: "Length/size arguments are invalid.",
    message: "ifft2: invalid transform length argument",
};

pub const IFFT2_ERROR_INVALID_SIZE_VECTOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INVALID_SIZE_VECTOR",
    identifier: Some("RunMat:ifft2:InvalidSizeVector"),
    when: "Single SIZE argument is invalid.",
    message: "ifft2: invalid size vector argument",
};

pub const IFFT2_ERROR_INVALID_SYMFLAG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INVALID_SYMFLAG",
    identifier: Some("RunMat:ifft2:InvalidSymflag"),
    when: "Symmetry flag is invalid or appears in an invalid position.",
    message: "ifft2: invalid symmetry flag usage",
};

pub const IFFT2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INVALID_INPUT",
    identifier: Some("RunMat:ifft2:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "ifft2: invalid input",
};

pub const IFFT2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INTERNAL",
    identifier: Some("RunMat:ifft2:Internal"),
    when: "IFFT2 execution or tensor shaping fails.",
    message: "ifft2: internal error",
};
pub const IFFT2_ERROR_PROVIDER_INTEGRITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.PROVIDER_INTEGRITY",
    identifier: Some("RunMat:ifft2:ProviderIntegrity"),
    when:
        "The provider returns ownership, shape, or physical metadata inconsistent with the request.",
    message: "ifft2: provider integrity error",
};

pub const IFFT2_ERRORS: [BuiltinErrorDescriptor; 7] = [
    IFFT2_ERROR_ARG_COUNT,
    IFFT2_ERROR_INVALID_LENGTH,
    IFFT2_ERROR_INVALID_SIZE_VECTOR,
    IFFT2_ERROR_INVALID_SYMFLAG,
    IFFT2_ERROR_INVALID_INPUT,
    IFFT2_ERROR_INTERNAL,
    IFFT2_ERROR_PROVIDER_INTEGRITY,
];

pub const IFFT2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IFFT2_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IFFT2_ERRORS,
};
