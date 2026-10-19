use crate::*;

pub const RANK_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rank-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rank with a typed-integer matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RankIntegerInputExtension"),
};

pub const RANK_INTEGER_TOLERANCE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "rank-integer-tolerance",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "rank with a typed-integer tolerance is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RankIntegerToleranceExtension"),
    };
pub const RANK_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rank-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rank with a logical matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RankLogicalInputExtension"),
};
pub const RANK_LOGICAL_TOLERANCE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "rank-logical-tolerance",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "rank with a logical tolerance is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RankLogicalToleranceExtension"),
    };

pub const RANK_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    RANK_INTEGER_INPUT_EXTENSION,
    RANK_INTEGER_TOLERANCE_EXTENSION,
    RANK_LOGICAL_INPUT_EXTENSION,
    RANK_LOGICAL_TOLERANCE_EXTENSION,
];

pub const RANK_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public matrix classes are single and double; RunMat mode admits real integer matrices only after an exact binary64-boundary check.",
    }];

pub const RANK_INTEGER_TOLERANCE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "tol",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public tolerance is a real scalar without an integer class list; RunMat mode admits typed integer tolerance values only when exactly representable as binary64.",
    }];

pub const RANK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "k = rank(integer_A, tol?)",
        inputs: &RANK_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The gated integer matrix crosses one checked binary64 SVD boundary; the structural rank count is returned as an exactly representable double scalar and automatic residency may restore it to the owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "k = rank(A, integer_tol)",
        inputs: &RANK_INTEGER_TOLERANCE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed integer tolerance is a gated RunMat extension and crosses into the SVD threshold only after exact binary64 representability and nonnegative finite validation.",
    },
];

pub const RANK_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Estimated matrix rank.",
}];

pub const RANK_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

pub const RANK_INPUTS_TOL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Singular-value threshold.",
    },
];

pub const RANK_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "k = rank(A)",
        inputs: &RANK_INPUTS,
        outputs: &RANK_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "k = rank(A, tol)",
        inputs: &RANK_INPUTS_TOL,
        outputs: &RANK_OUTPUT,
    },
];

pub const RANK_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RANK.INVALID_ARGUMENT",
    identifier: Some("RunMat:rank:InvalidArgument"),
    when: "Optional tolerance argument is malformed or outside accepted bounds.",
    message: "rank: invalid argument",
};

pub const RANK_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RANK.INVALID_INPUT",
    identifier: Some("RunMat:rank:InvalidInput"),
    when: "Input shape/type cannot be processed for rank evaluation.",
    message: "rank: invalid input",
};

pub const RANK_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RANK.INTERNAL",
    identifier: Some("RunMat:rank:Internal"),
    when: "Runtime fails while computing rank or executing fallback/upload paths.",
    message: "rank: internal runtime failure",
};

pub const RANK_ERRORS: [BuiltinErrorDescriptor; 3] = [
    RANK_ERROR_INVALID_ARGUMENT,
    RANK_ERROR_INVALID_INPUT,
    RANK_ERROR_INTERNAL,
];

pub const RANK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RANK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RANK_ERRORS,
};
