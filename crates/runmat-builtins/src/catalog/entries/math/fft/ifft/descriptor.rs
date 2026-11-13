use crate::*;

pub const IFFT_WIDE_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifft-wide-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifft with host int64 or uint64 data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftWideIntegerDataExtension"),
};

pub const IFFT_WIDE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifft-wide-integer-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifft with int64 or uint64 N or DIM controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftWideIntegerControlExtension"),
};

pub const IFFT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [IFFT_WIDE_DATA_EXTENSION, IFFT_WIDE_CONTROL_EXTENSION];

pub const IFFT_INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::catalog::integer::INTEGER_CLASSES_THROUGH_32_BITS,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "Documented integer data enters the floating inverse-FFT domain and produces double output.",
}];

pub const IFFT_WIDE_INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &[BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Host wide integers are gated and must be exactly representable as double; resident wide integers reject before gather.",
    },
];

pub const IFFT_INTEGER_CONTROLS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "N or DIM",
        classes: &crate::catalog::integer::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "N and DIM are parsed as exact structural controls.",
    }];

pub const IFFT_WIDE_INTEGER_CONTROLS: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "N or DIM",
        classes: &[BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "Wide structural controls are independently gated and decoded from authoritative integer storage.",
    },
];

pub const IFFT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifft(integer_X, ...)",
        inputs: &IFFT_INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented integer data crosses once into double; symmetric output is real and nonsymmetric output is complex.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifft(X, integer_N, integer_DIM, ...)",
        inputs: &IFFT_INTEGER_CONTROLS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Documented controls are exact and are classified before provider access.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifft(int64_or_uint64_X, ...)",
        inputs: &IFFT_WIDE_INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only wide data cannot silently round at the double boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifft(X, int64_or_uint64_N_or_DIM, ...)",
        inputs: &IFFT_WIDE_INTEGER_CONTROLS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only wide controls are independently gated and parsed exactly.",
    },
];

pub const IFFT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Inverse FFT result.",
}];

pub const IFFT_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input spectrum or signal.",
}];

pub const IFFT_INPUTS_WITH_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Transform length along selected dimension.",
    },
];

pub const IFFT_INPUTS_WITH_SYMFLAG: [BuiltinParamDescriptor; 2] = [
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

pub const IFFT_INPUTS_WITH_N_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
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

pub const IFFT_INPUTS_WITH_N_SYMFLAG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Transform length along selected dimension.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

pub const IFFT_INPUTS_WITH_N_DIM_SYMFLAG: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
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
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

pub const IFFT_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "Y = ifft(X)",
        inputs: &IFFT_INPUTS_CORE,
        outputs: &IFFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft(X, N)",
        inputs: &IFFT_INPUTS_WITH_N,
        outputs: &IFFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft(X, symflag)",
        inputs: &IFFT_INPUTS_WITH_SYMFLAG,
        outputs: &IFFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft(X, N, DIM)",
        inputs: &IFFT_INPUTS_WITH_N_DIM,
        outputs: &IFFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft(X, N, symflag)",
        inputs: &IFFT_INPUTS_WITH_N_SYMFLAG,
        outputs: &IFFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft(X, N, DIM, symflag)",
        inputs: &IFFT_INPUTS_WITH_N_DIM_SYMFLAG,
        outputs: &IFFT_OUTPUT,
    },
];

pub const IFFT_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT.ARG_COUNT",
    identifier: Some("RunMat:ifft:ArgCount"),
    when: "More than four input arguments are supplied.",
    message: "ifft: invalid argument count",
};

pub const IFFT_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT.INVALID_LENGTH",
    identifier: Some("RunMat:ifft:InvalidLength"),
    when: "Length argument N is invalid.",
    message: "ifft: invalid length argument",
};

pub const IFFT_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT.INVALID_DIMENSION",
    identifier: Some("RunMat:ifft:InvalidDimension"),
    when: "Dimension argument DIM is invalid.",
    message: "ifft: invalid dimension argument",
};

pub const IFFT_ERROR_INVALID_SYMFLAG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT.INVALID_SYMFLAG",
    identifier: Some("RunMat:ifft:InvalidSymflag"),
    when: "Symmetry flag is invalid or appears in an invalid position.",
    message: "ifft: invalid symmetry flag usage",
};

pub const IFFT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT.INVALID_INPUT",
    identifier: Some("RunMat:ifft:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "ifft: invalid input",
};

pub const IFFT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT.INTERNAL",
    identifier: Some("RunMat:ifft:Internal"),
    when: "IFFT execution or tensor shaping fails.",
    message: "ifft: internal error",
};
pub const IFFT_ERROR_PROVIDER_INTEGRITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT.PROVIDER_INTEGRITY",
    identifier: Some("RunMat:ifft:ProviderIntegrity"),
    when:
        "The provider returns ownership, shape, or physical metadata inconsistent with the request.",
    message: "ifft: provider integrity error",
};

pub const IFFT_ERRORS: [BuiltinErrorDescriptor; 7] = [
    IFFT_ERROR_ARG_COUNT,
    IFFT_ERROR_INVALID_LENGTH,
    IFFT_ERROR_INVALID_DIMENSION,
    IFFT_ERROR_INVALID_SYMFLAG,
    IFFT_ERROR_INVALID_INPUT,
    IFFT_ERROR_INTERNAL,
    IFFT_ERROR_PROVIDER_INTEGRITY,
];

pub const IFFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IFFT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IFFT_ERRORS,
};
