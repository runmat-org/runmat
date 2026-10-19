use crate::*;

pub const IFFTN_WIDE_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifftn-wide-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifftn with host int64 or uint64 data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftnWideIntegerDataExtension"),
};
pub const IFFTN_WIDE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifftn-wide-integer-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifftn with int64 or uint64 SIZE controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftnWideIntegerControlExtension"),
};
pub const IFFTN_SHORT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifftn-short-size-vector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifftn SIZE vectors shorter than ndims(X) are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftnShortSizeExtension"),
};
pub const IFFTN_ZERO_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifftn-zero-size-vector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifftn SIZE vectors containing zero are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftnZeroSizeExtension"),
};
pub const IFFTN_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    IFFTN_WIDE_DATA_EXTENSION,
    IFFTN_WIDE_CONTROL_EXTENSION,
    IFFTN_SHORT_SIZE_EXTENSION,
    IFFTN_ZERO_SIZE_EXTENSION,
];

pub const IFFTN_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::catalog::integer::INTEGER_CLASSES_THROUGH_32_BITS,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Documented integer arrays enter the double N-dimensional inverse-FFT domain.",
}];
pub const IFFTN_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "SIZE",
    classes: &crate::catalog::integer::INTEGER_CLASSES_THROUGH_32_BITS,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "SIZE is a positive integer vector with length at least ndims(X); logical vectors are documented.",
}];
pub const IFFTN_WIDE_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &[BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Host wide data is gated and must be exactly representable as double.",
    }];
pub const IFFTN_WIDE_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "SIZE",
        classes: &[BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "Wide SIZE controls are independently gated and decoded exactly.",
    }];
pub const IFFTN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifftn(integer_X, ...)",
        inputs: &IFFTN_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented integer data produces double output; symmetric changes complexity only.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifftn(X, integer_SIZE, ...)",
        inputs: &IFFTN_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Documented SIZE is decoded exactly and validated before execution.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifftn(int64_or_uint64_X, ...)",
        inputs: &IFFTN_WIDE_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only wide host data may not silently round; resident wide data rejects before gather.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifftn(X, int64_or_uint64_SIZE, ...)",
        inputs: &IFFTN_WIDE_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only wide SIZE controls are independently gated; short and zero vectors have separate gates.",
    },
];

pub const IFFTN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "N-D inverse FFT output.",
}];

pub const IFFTN_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input spectrum or signal.",
}];

pub const IFFTN_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [
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
        description: "Transform sizes per dimension.",
    },
];

pub const IFFTN_INPUTS_SYMFLAG: [BuiltinParamDescriptor; 2] = [
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

pub const IFFTN_INPUTS_SIZE_SYMFLAG: [BuiltinParamDescriptor; 3] = [
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
        description: "Transform sizes per dimension.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

pub const IFFTN_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "Y = ifftn(X)",
        inputs: &IFFTN_INPUTS_CORE,
        outputs: &IFFTN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifftn(X, SIZE)",
        inputs: &IFFTN_INPUTS_SIZE,
        outputs: &IFFTN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifftn(X, symflag)",
        inputs: &IFFTN_INPUTS_SYMFLAG,
        outputs: &IFFTN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifftn(X, SIZE, symflag)",
        inputs: &IFFTN_INPUTS_SIZE_SYMFLAG,
        outputs: &IFFTN_OUTPUT,
    },
];

pub const IFFTN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.ARG_COUNT",
    identifier: Some("RunMat:ifftn:ArgCount"),
    when: "More than three input arguments are supplied.",
    message: "ifftn: invalid argument count",
};

pub const IFFTN_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.INVALID_SIZE",
    identifier: Some("RunMat:ifftn:InvalidSize"),
    when: "SIZE argument is invalid.",
    message: "ifftn: invalid SIZE argument",
};

pub const IFFTN_ERROR_INVALID_SYMFLAG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.INVALID_SYMFLAG",
    identifier: Some("RunMat:ifftn:InvalidSymflag"),
    when: "Symmetry flag is invalid or appears in an invalid position.",
    message: "ifftn: invalid symmetry flag usage",
};

pub const IFFTN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.INVALID_INPUT",
    identifier: Some("RunMat:ifftn:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "ifftn: invalid input",
};

pub const IFFTN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.INTERNAL",
    identifier: Some("RunMat:ifftn:Internal"),
    when: "IFFTN execution or tensor shaping fails.",
    message: "ifftn: internal error",
};
pub const IFFTN_ERROR_PROVIDER_INTEGRITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.PROVIDER_INTEGRITY",
    identifier: Some("RunMat:ifftn:ProviderIntegrity"),
    when:
        "The provider returns ownership, shape, or physical metadata inconsistent with the request.",
    message: "ifftn: provider integrity error",
};

pub const IFFTN_ERRORS: [BuiltinErrorDescriptor; 6] = [
    IFFTN_ERROR_ARG_COUNT,
    IFFTN_ERROR_INVALID_SIZE,
    IFFTN_ERROR_INVALID_SYMFLAG,
    IFFTN_ERROR_INVALID_INPUT,
    IFFTN_ERROR_INTERNAL,
    IFFTN_ERROR_PROVIDER_INTEGRITY,
];

pub const IFFTN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IFFTN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IFFTN_ERRORS,
};
