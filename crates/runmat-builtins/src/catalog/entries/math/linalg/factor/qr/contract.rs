use crate::*;

pub const QR_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qr-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qr with a typed-integer matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QrIntegerInputExtension"),
};

pub const QR_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qr-integer-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qr with a typed-integer economy selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QrIntegerOptionExtension"),
};
pub const QR_RESIDENT_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qr-resident-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qr with a gpuArray option value is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QrResidentOptionExtension"),
};
pub const QR_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qr-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qr with a logical matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QrLogicalInputExtension"),
};
pub const QR_LOGICAL_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qr-logical-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qr with a logical economy selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QrLogicalOptionExtension"),
};

pub const QR_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    QR_INTEGER_INPUT_EXTENSION,
    QR_INTEGER_OPTION_EXTENSION,
    QR_RESIDENT_OPTION_EXTENSION,
    QR_LOGICAL_INPUT_EXTENSION,
    QR_LOGICAL_OPTION_EXTENSION,
];

pub const QR_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The public full-matrix classes are single and double; RunMat mode admits real integer matrices only after an exact binary64-boundary check.",
}];

pub const QR_INTEGER_OPTION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "economy_selector",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented legacy economy selector is the numeric literal 0; RunMat mode additionally accepts typed integer scalar zero without a floating conversion.",
    }];

pub const QR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[Q, R, P] = qr(integer_A, options...)",
        inputs: &QR_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The gated matrix crosses one checked binary64 QR boundary. Q, R, and matrix-form P are double; vector-form P is a one-based double index vector. Exact-owner restoration preserves these classes and input provenance through checked uploads; unsupported transfers report an error. Only requested outputs are materialized.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[Q, R, P] = qr(A, integer_zero)",
        inputs: &QR_INTEGER_OPTION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed integer scalar zero selects the legacy economy/vector form exactly in RunMat mode; every nonzero or nonscalar integer option rejects as an invalid option, and an explicit resident option is independently gated before exact-owner download.",
    },
];

pub const QR_OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Upper triangular (or trapezoidal) factor.",
}];

pub const QR_OUTPUT_QR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Q",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Orthogonal/unitary factor.",
    },
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper triangular (or trapezoidal) factor.",
    },
];

pub const QR_OUTPUT_QRP: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "Q",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Orthogonal/unitary factor.",
    },
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper triangular (or trapezoidal) factor.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Permutation matrix or vector based on pivot mode.",
    },
];

pub const QR_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix to factorize.",
}];

pub const QR_INPUTS_A_OPTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to factorize.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option token (`0`, `econ`, `matrix`, or `vector`).",
    },
];

pub const QR_INPUTS_A_OPTION_OPTION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to factorize.",
    },
    BuiltinParamDescriptor {
        name: "option1",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First option token (`0`, `econ`, `matrix`, or `vector`).",
    },
    BuiltinParamDescriptor {
        name: "option2",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second option token (`0`, `econ`, `matrix`, or `vector`).",
    },
];

const SPARSE_S: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse single or double coefficient matrix.",
};
const SPARSE_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense or sparse floating right-hand sides with the same row count as S.",
};
const SPARSE_OPTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Option token (0, econ, matrix, or vector).",
};
const SPARSE_OPTION2: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "option2",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second option token (0, econ, matrix, or vector).",
};
const SPARSE_C: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Dense Q' * B, truncated to the economy height when selected; Q is not materialized.",
};
const SPARSE_R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse upper triangular or trapezoidal factor in the coefficient class.",
};
const SPARSE_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "P",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse double permutation matrix or dense one-based double column index vector.",
};
pub const QR_INPUTS_SB: [BuiltinParamDescriptor; 2] = [SPARSE_S, SPARSE_B];
pub const QR_INPUTS_SB_OPTION: [BuiltinParamDescriptor; 3] = [SPARSE_S, SPARSE_B, SPARSE_OPTION];
pub const QR_INPUTS_SB_OPTIONS: [BuiltinParamDescriptor; 4] =
    [SPARSE_S, SPARSE_B, SPARSE_OPTION, SPARSE_OPTION2];
pub const QR_OUTPUT_CR: [BuiltinParamDescriptor; 2] = [SPARSE_C, SPARSE_R];
pub const QR_OUTPUT_CRP: [BuiltinParamDescriptor; 3] = [SPARSE_C, SPARSE_R, SPARSE_P];

pub const QR_SIGNATURES: [BuiltinSignatureDescriptor; 15] = [
    BuiltinSignatureDescriptor {
        label: "R = qr(A)",
        inputs: &QR_INPUTS_A,
        outputs: &QR_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = qr(A, option)",
        inputs: &QR_INPUTS_A_OPTION,
        outputs: &QR_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = qr(A, option1, option2)",
        inputs: &QR_INPUTS_A_OPTION_OPTION,
        outputs: &QR_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R] = qr(A)",
        inputs: &QR_INPUTS_A,
        outputs: &QR_OUTPUT_QR,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R] = qr(A, option)",
        inputs: &QR_INPUTS_A_OPTION,
        outputs: &QR_OUTPUT_QR,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R] = qr(A, option1, option2)",
        inputs: &QR_INPUTS_A_OPTION_OPTION,
        outputs: &QR_OUTPUT_QR,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R, P] = qr(A)",
        inputs: &QR_INPUTS_A,
        outputs: &QR_OUTPUT_QRP,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R, P] = qr(A, option)",
        inputs: &QR_INPUTS_A_OPTION,
        outputs: &QR_OUTPUT_QRP,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R, P] = qr(A, option1, option2)",
        inputs: &QR_INPUTS_A_OPTION_OPTION,
        outputs: &QR_OUTPUT_QRP,
    },
    BuiltinSignatureDescriptor {
        label: "[C, R] = qr(S, B)",
        inputs: &QR_INPUTS_SB,
        outputs: &QR_OUTPUT_CR,
    },
    BuiltinSignatureDescriptor {
        label: "[C, R] = qr(S, B, option)",
        inputs: &QR_INPUTS_SB_OPTION,
        outputs: &QR_OUTPUT_CR,
    },
    BuiltinSignatureDescriptor {
        label: "[C, R] = qr(S, B, option1, option2)",
        inputs: &QR_INPUTS_SB_OPTIONS,
        outputs: &QR_OUTPUT_CR,
    },
    BuiltinSignatureDescriptor {
        label: "[C, R, P] = qr(S, B)",
        inputs: &QR_INPUTS_SB,
        outputs: &QR_OUTPUT_CRP,
    },
    BuiltinSignatureDescriptor {
        label: "[C, R, P] = qr(S, B, option)",
        inputs: &QR_INPUTS_SB_OPTION,
        outputs: &QR_OUTPUT_CRP,
    },
    BuiltinSignatureDescriptor {
        label: "[C, R, P] = qr(S, B, option1, option2)",
        inputs: &QR_INPUTS_SB_OPTIONS,
        outputs: &QR_OUTPUT_CRP,
    },
];

pub const QR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QR.INVALID_ARGUMENT",
    identifier: Some("RunMat:qr:InvalidArgument"),
    when: "Options are invalid or output count exceeds supported forms.",
    message: "qr currently supports at most three outputs",
};

pub const QR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QR.INVALID_INPUT",
    identifier: Some("RunMat:qr:InvalidInput"),
    when: "Input is non-numeric or has rank greater than two.",
    message: "qr: expected a numeric 2-D input matrix",
};

pub const QR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QR.INTERNAL",
    identifier: Some("RunMat:qr:Internal"),
    when: "QR runtime fails to materialize output values.",
    message: "qr: internal runtime failure",
};

pub const QR_ERRORS: [BuiltinErrorDescriptor; 3] = [
    QR_ERROR_INVALID_ARGUMENT,
    QR_ERROR_INVALID_INPUT,
    QR_ERROR_INTERNAL,
];

pub const QR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &QR_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &QR_ERRORS,
};
