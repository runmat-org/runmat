use crate::*;

pub const FFTSHIFT_MULTI_DIM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fftshift-multi-dimension-selector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fftshift numeric dimension vectors and logical masks are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FftshiftMultiDimensionExtension"),
};
pub const FFTSHIFT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [FFTSHIFT_MULTI_DIM_EXTENSION];

pub const FFTSHIFT_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "All integer classes are reordered exactly with class, shape, and complexity preserved.",
    },
];
pub const FFTSHIFT_DIM_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "DIM",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented DIM form is one positive scalar parsed from authoritative integer storage.",
    },
];
pub const FFTSHIFT_MULTI_DIM_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "DIMS",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "RunMat-only numeric vectors are decoded exactly; logical mask vectors share the same independent extension gate.",
    },
];
pub const FFTSHIFT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fftshift(integer_X)",
        inputs: &FFTSHIFT_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Real and typed-complex integer values are reordered exactly with class preserved; resident fallback re-uploads to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fftshift(X, integer_DIM)",
        inputs: &FFTSHIFT_DIM_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "All eight documented integer scalar classes select one dimension exactly.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = fftshift(X, integer_DIMS)",
        inputs: &FFTSHIFT_MULTI_DIM_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only multi-axis vector/mask selection is independently gated from documented scalar DIM semantics.",
    },
];

pub const FFTSHIFT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Shifted array with the same size and type family as X.",
}];

pub const FFTSHIFT_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, complex, logical, or gpuArray input.",
}];

pub const FFTSHIFT_INPUTS_DIMS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, complex, logical, or gpuArray input.",
    },
    BuiltinParamDescriptor {
        name: "DIM",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Positive scalar dimension; vectors and logical masks are RunMat extensions.",
    },
];

pub const FFTSHIFT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = fftshift(X)",
        inputs: &FFTSHIFT_INPUTS_CORE,
        outputs: &FFTSHIFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fftshift(X, DIM)",
        inputs: &FFTSHIFT_INPUTS_DIMS,
        outputs: &FFTSHIFT_OUTPUT,
    },
];

pub const FFTSHIFT_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.ARG_COUNT",
    identifier: Some("RunMat:fftshift:ArgCount"),
    when: "More than two input arguments are supplied.",
    message: "fftshift: invalid argument count",
};

pub const FFTSHIFT_ERROR_INVALID_DIMS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.INVALID_DIMS",
    identifier: Some("RunMat:fftshift:InvalidDimensions"),
    when: "DIM argument is malformed or out of range.",
    message: "fftshift: invalid dimension argument",
};

pub const FFTSHIFT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.INVALID_INPUT",
    identifier: Some("RunMat:fftshift:InvalidInput"),
    when: "X is not a supported numeric/logical input type.",
    message: "fftshift: expected numeric or logical input",
};

pub const FFTSHIFT_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:fftshift:UnsupportedInput"),
    when: "X is an unsupported object/cell/function/meta runtime type.",
    message: "fftshift: unsupported input type",
};

pub const FFTSHIFT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.INTERNAL",
    identifier: Some("RunMat:fftshift:Internal"),
    when: "Shifting, tensor reconstruction, or GPU transfer operations fail.",
    message: "fftshift: internal error",
};

pub const FFTSHIFT_ERROR_PROVIDER_INTEGRITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.PROVIDER_INTEGRITY",
    identifier: Some("RunMat:fftshift:ProviderIntegrity"),
    when: "An admitted circshift hook fails or returns an invalid, aliased, or wrong-owner result.",
    message: "fftshift: provider integrity error",
};

pub const FFTSHIFT_ERRORS: [BuiltinErrorDescriptor; 6] = [
    FFTSHIFT_ERROR_ARG_COUNT,
    FFTSHIFT_ERROR_INVALID_DIMS,
    FFTSHIFT_ERROR_INVALID_INPUT,
    FFTSHIFT_ERROR_UNSUPPORTED_INPUT,
    FFTSHIFT_ERROR_INTERNAL,
    FFTSHIFT_ERROR_PROVIDER_INTEGRITY,
];

pub const FFTSHIFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FFTSHIFT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FFTSHIFT_ERRORS,
};
