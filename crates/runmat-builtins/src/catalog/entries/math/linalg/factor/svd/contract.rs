use crate::*;

pub const SVD_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "svd-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow exactly representable typed-integer matrices in svd",
    error_identifier: Some("RunMat:compatibility:SvdIntegerInputExtension"),
};
pub const SVD_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "svd-integer-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow a typed-integer zero economy selector in svd",
    error_identifier: Some("RunMat:compatibility:SvdIntegerOptionExtension"),
};
pub const SVD_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "svd-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow logical matrices in svd",
    error_identifier: Some("RunMat:compatibility:SvdLogicalInputExtension"),
};
pub const SVD_LOGICAL_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "svd-logical-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow a logical economy selector in svd",
    error_identifier: Some("RunMat:compatibility:SvdLogicalOptionExtension"),
};
pub const SVD_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    SVD_INTEGER_INPUT_EXTENSION,
    SVD_INTEGER_OPTION_EXTENSION,
    SVD_LOGICAL_INPUT_EXTENSION,
    SVD_LOGICAL_OPTION_EXTENSION,
];
pub const SVD_INTEGER_MATRIX: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public dense matrix classes are single and double; RunMat mode admits real integer matrices only after an exact binary64-boundary check.",
    }];
pub const SVD_INTEGER_OPTION: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "economy selector",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode accepts a typed integer scalar only when its exact value is zero.",
    }];
pub const SVD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[U, S, V] = svd(integer_A, options...)",
        inputs: &SVD_INTEGER_MATRIX,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The gated matrix crosses one checked binary64 SVD boundary. Values not exactly representable as double reject before factorization or provider dispatch; outputs are double and follow the requested full/economy/vector form.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "S = svd(A, integer_zero)",
        inputs: &SVD_INTEGER_OPTION,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The exact gated zero selects economy size without floating conversion; nonzero or nonscalar integer options reject.",
    },
];

pub const SVD_OUTPUT_S: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Singular values (vector by default).",
}];

pub const SVD_OUTPUT_US: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left singular vectors.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal singular-value matrix or vector based on options.",
    },
];

pub const SVD_OUTPUT_USV: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left singular vectors.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal singular-value matrix or vector based on options.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right singular vectors.",
    },
];

pub const SVD_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix to decompose.",
}];

pub const SVD_INPUTS_A_OPTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to decompose.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option token (`0`, `econ`, `full`, `vector`, or `matrix`).",
    },
];

pub const SVD_INPUTS_A_OPTION_OPTION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to decompose.",
    },
    BuiltinParamDescriptor {
        name: "option1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First option token (`0`, `econ`, `full`, `vector`, or `matrix`).",
    },
    BuiltinParamDescriptor {
        name: "option2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second option token (`0`, `econ`, `full`, `vector`, or `matrix`).",
    },
];

pub const SVD_SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "S = svd(A)",
        inputs: &SVD_INPUTS_A,
        outputs: &SVD_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "S = svd(A, option)",
        inputs: &SVD_INPUTS_A_OPTION,
        outputs: &SVD_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "S = svd(A, option1, option2)",
        inputs: &SVD_INPUTS_A_OPTION_OPTION,
        outputs: &SVD_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "[U, S] = svd(A)",
        inputs: &SVD_INPUTS_A,
        outputs: &SVD_OUTPUT_US,
    },
    BuiltinSignatureDescriptor {
        label: "[U, S] = svd(A, option)",
        inputs: &SVD_INPUTS_A_OPTION,
        outputs: &SVD_OUTPUT_US,
    },
    BuiltinSignatureDescriptor {
        label: "[U, S] = svd(A, option1, option2)",
        inputs: &SVD_INPUTS_A_OPTION_OPTION,
        outputs: &SVD_OUTPUT_US,
    },
    BuiltinSignatureDescriptor {
        label: "[U, S, V] = svd(A)",
        inputs: &SVD_INPUTS_A,
        outputs: &SVD_OUTPUT_USV,
    },
    BuiltinSignatureDescriptor {
        label: "[U, S, V] = svd(A, option)",
        inputs: &SVD_INPUTS_A_OPTION,
        outputs: &SVD_OUTPUT_USV,
    },
    BuiltinSignatureDescriptor {
        label: "[U, S, V] = svd(A, option1, option2)",
        inputs: &SVD_INPUTS_A_OPTION_OPTION,
        outputs: &SVD_OUTPUT_USV,
    },
];

pub const SVD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SVD.INVALID_ARGUMENT",
    identifier: Some("RunMat:svd:InvalidArgument"),
    when: "Option arguments or requested output count are invalid.",
    message: "svd currently supports at most three outputs",
};

pub const SVD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SVD.INVALID_INPUT",
    identifier: Some("RunMat:svd:InvalidInput"),
    when: "Input is unsupported or not a 2-D numeric/logical matrix.",
    message: "svd: expected numeric or logical values",
};

pub const SVD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SVD.INTERNAL",
    identifier: Some("RunMat:svd:Internal"),
    when: "Runtime cannot materialize SVD outputs.",
    message: "svd: internal runtime failure",
};

pub const SVD_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SVD_ERROR_INVALID_ARGUMENT,
    SVD_ERROR_INVALID_INPUT,
    SVD_ERROR_INTERNAL,
];

pub const SVD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SVD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SVD_ERRORS,
};
