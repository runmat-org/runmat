use crate::*;

pub const VECNORM_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "vecnorm-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "vecnorm with typed-integer array data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:VecnormIntegerInputExtension"),
    };
pub const VECNORM_INTEGER_ORDER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "vecnorm-integer-order",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "vecnorm with a typed-integer norm order is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:VecnormIntegerOrderExtension"),
    };
pub const VECNORM_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    VECNORM_INTEGER_INPUT_EXTENSION,
    VECNORM_INTEGER_ORDER_EXTENSION,
];

pub const VECNORM_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented data classes are single and double. RunMat mode accepts real integer arrays only when every value is exactly representable at the binary64 norm boundary.",
    }];
pub const VECNORM_INTEGER_ORDER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed norm order is a RunMat extension and enters the floating exponent only after exact binary64 representability is proved.",
    }];
pub const VECNORM_INTEGER_DIM_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented dimension accepts every integer class and is decoded exactly as a positive platform-sized structural value.",
    }];

pub const VECNORM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "N = vecnorm(integer_A, ___)",
        inputs: &VECNORM_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The gated integer array crosses one checked binary64 norm boundary. Host results are double and automatic residency restores results to the owning provider when supported.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "N = vecnorm(A, integer_p, dim?)",
        inputs: &VECNORM_INTEGER_ORDER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The gated typed p is checked before conversion into the norm exponent; it does not select the output class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "N = vecnorm(A, p, integer_dim)",
        inputs: &VECNORM_INTEGER_DIM_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Dimension selection remains exact and does not enter norm arithmetic; invalid or non-platform-sized values reject without floating conversion.",
    },
];

pub const VECNORM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Vector-wise norm values.",
}];

pub const VECNORM_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input numeric array.",
}];

pub const VECNORM_INPUTS_A_P: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input numeric array.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("2"),
        description: "Positive norm order or Inf.",
    },
];

pub const VECNORM_INPUTS_A_P_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input numeric array.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("2"),
        description: "Positive norm order or Inf.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Dimension to operate along.",
    },
];

pub const VECNORM_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "N = vecnorm(A)",
        inputs: &VECNORM_INPUTS_A,
        outputs: &VECNORM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "N = vecnorm(A, p)",
        inputs: &VECNORM_INPUTS_A_P,
        outputs: &VECNORM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "N = vecnorm(A, p, dim)",
        inputs: &VECNORM_INPUTS_A_P_DIM,
        outputs: &VECNORM_OUTPUT,
    },
];

pub const VECNORM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VECNORM.INVALID_ARGUMENT",
    identifier: Some("RunMat:vecnorm:InvalidArgument"),
    when: "The norm order or dimension argument is malformed or unsupported.",
    message: "vecnorm: invalid argument",
};

pub const VECNORM_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VECNORM.INVALID_INPUT",
    identifier: Some("RunMat:vecnorm:InvalidInput"),
    when: "Input values cannot be converted to a supported numeric array domain.",
    message: "vecnorm: invalid input",
};

pub const VECNORM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VECNORM.INTERNAL",
    identifier: Some("RunMat:vecnorm:Internal"),
    when: "Runtime fails while reducing, allocating, gathering, or uploading values.",
    message: "vecnorm: internal runtime failure",
};

pub const VECNORM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    VECNORM_ERROR_INVALID_ARGUMENT,
    VECNORM_ERROR_INVALID_INPUT,
    VECNORM_ERROR_INTERNAL,
];

pub const VECNORM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VECNORM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VECNORM_ERRORS,
};
