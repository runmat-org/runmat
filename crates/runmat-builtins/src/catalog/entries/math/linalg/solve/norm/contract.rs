use crate::*;

pub const NORM_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "norm-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "norm with native-class integer input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormIntegerDataExtension"),
};

pub const NORM_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "norm-logical-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "norm with logical input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormLogicalDataExtension"),
};

pub const NORM_LOGICAL_ORDER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "norm-logical-order",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "norm with a logical norm-order selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormLogicalOrderExtension"),
};

pub const NORM_ZERO_ORDER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "norm-zero-order",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "norm with the undocumented zero-norm selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormZeroOrderExtension"),
};

pub const NORM_NUCLEAR_ORDER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "norm-nuclear-order",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "norm with the undocumented nuclear-norm selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormNuclearOrderExtension"),
};

pub const NORM_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    NORM_INTEGER_DATA_EXTENSION,
    NORM_LOGICAL_DATA_EXTENSION,
    NORM_LOGICAL_ORDER_EXTENSION,
    NORM_ZERO_ORDER_EXTENSION,
    NORM_NUCLEAR_ORDER_EXTENSION,
];

pub const NORM_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Native integer vectors and matrices are gated by norm-integer-data and enter the explicitly floating norm domain from authoritative storage.",
    }];

pub const NORM_INTEGER_ORDER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented positive real scalar norm order accepts exact native integer scalars; wide values must be representable at the floating exponent boundary.",
    }];

pub const NORM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "n = norm(integer_A, p)",
        inputs: &NORM_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer A is a RunMat-only extension. Host and automatic-resident inputs materialize once into the floating norm domain; restoration uses the exact owner only when it can preserve double, while incompatible automatic residency may remain host and explicit residency errors.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "n = norm(A, integer_p)",
        inputs: &NORM_INTEGER_ORDER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer p is parsed exactly before conversion to the documented real scalar selector; GPU-resident order selectors remain unsupported.",
    },
];

pub const NORM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Norm value of A.",
}];

pub const NORM_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input vector or matrix.",
}];

pub const NORM_INPUTS_ORDER: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Norm order selector.",
    },
];

pub const NORM_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "n = norm(A)",
        inputs: &NORM_INPUTS,
        outputs: &NORM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "n = norm(A, p)",
        inputs: &NORM_INPUTS_ORDER,
        outputs: &NORM_OUTPUT,
    },
];

pub const NORM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORM.INVALID_ARGUMENT",
    identifier: Some("RunMat:norm:InvalidArgument"),
    when: "Norm-order argument is malformed or unsupported for the requested domain.",
    message: "norm: invalid argument",
};

pub const NORM_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORM.INVALID_INPUT",
    identifier: Some("RunMat:norm:InvalidInput"),
    when: "Input shape/type cannot be processed for norm evaluation.",
    message: "norm: invalid input",
};

pub const NORM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORM.INTERNAL",
    identifier: Some("RunMat:norm:Internal"),
    when: "Runtime fails while computing norm or executing fallback/upload paths.",
    message: "norm: internal runtime failure",
};

pub const NORM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    NORM_ERROR_INVALID_ARGUMENT,
    NORM_ERROR_INVALID_INPUT,
    NORM_ERROR_INTERNAL,
];

pub const NORM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NORM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NORM_ERRORS,
};
