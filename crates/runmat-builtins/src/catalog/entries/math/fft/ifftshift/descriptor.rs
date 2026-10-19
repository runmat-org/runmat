use crate::*;

pub const IFFTSHIFT_MULTI_DIM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifftshift-multi-dimension-selector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifftshift numeric dimension vectors and logical masks are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftshiftMultiDimensionExtension"),
};
pub const IFFTSHIFT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [IFFTSHIFT_MULTI_DIM_EXTENSION];

pub const IFFTSHIFT_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "All integer classes are reordered exactly with class, shape, and complexity preserved.",
    },
];
pub const IFFTSHIFT_DIM_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "DIM",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented DIM form is one positive scalar parsed from authoritative storage.",
    }];
pub const IFFTSHIFT_MULTI_DIM_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "DIMS",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes:
            "RunMat-only numeric vectors are decoded exactly; logical masks share the same gate.",
    }];
pub const IFFTSHIFT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifftshift(integer_X)",
        inputs: &IFFTSHIFT_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Real and typed-complex integer values are reordered exactly with class preserved; resident fallback restores to the owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifftshift(X, integer_DIM)",
        inputs: &IFFTSHIFT_DIM_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "All eight documented integer scalar classes select one dimension exactly.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = ifftshift(X, integer_DIMS)",
        inputs: &IFFTSHIFT_MULTI_DIM_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only multi-axis vector/mask selection is independently gated.",
    },
];

pub const IFFTSHIFT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Shifted array with the same size and type family as X.",
}];

pub const IFFTSHIFT_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, complex, logical, or gpuArray input.",
}];

pub const IFFTSHIFT_INPUTS_DIMS: [BuiltinParamDescriptor; 2] = [
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
        default: Some("[]"),
        description: "Dimension selector (scalar, numeric vector, or logical mask vector).",
    },
];

pub const IFFTSHIFT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = ifftshift(X)",
        inputs: &IFFTSHIFT_INPUTS_CORE,
        outputs: &IFFTSHIFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifftshift(X, DIM)",
        inputs: &IFFTSHIFT_INPUTS_DIMS,
        outputs: &IFFTSHIFT_OUTPUT,
    },
];

pub const IFFTSHIFT_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTSHIFT.ARG_COUNT",
    identifier: Some("RunMat:ifftshift:ArgCount"),
    when: "More than two input arguments are supplied.",
    message: "ifftshift: invalid argument count",
};

pub const IFFTSHIFT_ERROR_INVALID_DIMS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTSHIFT.INVALID_DIMS",
    identifier: Some("RunMat:ifftshift:InvalidDimensions"),
    when: "DIM argument is malformed or out of range.",
    message: "ifftshift: invalid dimension argument",
};

pub const IFFTSHIFT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTSHIFT.INVALID_INPUT",
    identifier: Some("RunMat:ifftshift:InvalidInput"),
    when: "X is not a supported numeric/logical input type.",
    message: "ifftshift: expected numeric or logical input",
};

pub const IFFTSHIFT_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTSHIFT.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:ifftshift:UnsupportedInput"),
    when: "X is an unsupported object/cell/function/meta runtime type.",
    message: "ifftshift: unsupported input type",
};

pub const IFFTSHIFT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTSHIFT.INTERNAL",
    identifier: Some("RunMat:ifftshift:Internal"),
    when: "Shifting, tensor reconstruction, or GPU transfer operations fail.",
    message: "ifftshift: internal error",
};
pub const IFFTSHIFT_ERROR_PROVIDER_INTEGRITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTSHIFT.PROVIDER_INTEGRITY",
    identifier: Some("RunMat:ifftshift:ProviderIntegrity"),
    when:
        "The provider returns ownership, shape, or physical metadata inconsistent with the request.",
    message: "ifftshift: provider integrity error",
};

pub const IFFTSHIFT_ERRORS: [BuiltinErrorDescriptor; 6] = [
    IFFTSHIFT_ERROR_ARG_COUNT,
    IFFTSHIFT_ERROR_INVALID_DIMS,
    IFFTSHIFT_ERROR_INVALID_INPUT,
    IFFTSHIFT_ERROR_UNSUPPORTED_INPUT,
    IFFTSHIFT_ERROR_INTERNAL,
    IFFTSHIFT_ERROR_PROVIDER_INTEGRITY,
];

pub const IFFTSHIFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IFFTSHIFT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IFFTSHIFT_ERRORS,
};
