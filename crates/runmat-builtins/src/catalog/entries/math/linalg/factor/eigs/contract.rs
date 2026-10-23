use crate::*;

pub const EIGS_NONFLOATING_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "eigs-nonfloating-matrix",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "eigs with a non-single/non-double coefficient matrix is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EigsNonfloatingMatrixExtension"),
    };

pub const EIGS_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "eigs-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "eigs with a gpuArray argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EigsGpuInputExtension"),
};

pub const EIGS_INTEGER_SIGMA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "eigs-integer-sigma",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "eigs with a typed-integer sigma is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EigsIntegerSigmaExtension"),
};

pub const EIGS_INTEGER_START_VECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "eigs-integer-start-vector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "eigs with a typed-integer StartVector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EigsIntegerStartVectorExtension"),
    };

pub const EIGS_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    EIGS_NONFLOATING_MATRIX_EXTENSION,
    EIGS_GPU_INPUT_EXTENSION,
    EIGS_INTEGER_SIGMA_EXTENSION,
    EIGS_INTEGER_START_VECTOR_EXTENSION,
];

pub const EIGS_INTEGER_COEFFICIENT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Primary coefficient matrix.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Optional generalized coefficient matrix.",
    },
];
pub const EIGS_INTEGER_K_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "k",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Positive structural count parsed exactly in every supported integer class.",
    }];
pub const EIGS_INTEGER_SIGMA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sigma",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat-only numeric shift crossing an exact binary64 boundary.",
    }];
pub const EIGS_INTEGER_START_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "StartVector",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat-only initial-vector class. Sparse/callable iteration consumes exactly representable values; the separate dense route only validates this control.",
    }];

pub const EIGS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "eigs(A, B, ...) with a typed-integer coefficient",
        inputs: &EIGS_INTEGER_COEFFICIENT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only coefficients are checked before dense binary64 fallback.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "eigs(A, k, ...) with typed-integer k",
        inputs: &EIGS_INTEGER_K_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Documented exact structural control.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "eigs(A, k, sigma, ...) with typed-integer sigma",
        inputs: &EIGS_INTEGER_SIGMA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "RunMat-only shift extension.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "eigs(..., 'StartVector', v) with typed-integer v",
        inputs: &EIGS_INTEGER_START_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "RunMat-only iterative-control class extension.",
    },
];

const OUTPUT_D: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "d",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected eigenvalues as a column vector.",
}];

const OUTPUT_VD: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Selected right eigenvectors.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal matrix of selected eigenvalues.",
    },
];

const OUTPUT_VDFLAG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Selected right eigenvectors.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal matrix of selected eigenvalues.",
    },
    BuiltinParamDescriptor {
        name: "flag",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Convergence flag: 0 when all selected eigenpairs converged, 1 otherwise. Successful exact dense fallback returns 0.",
    },
];

const INPUT_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square numeric, sparse, logical, complex, or gpuArray matrix.",
}];

const INPUT_A_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Square matrix or selected callable operator Afun; callable form requires order n before other arguments.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional B, k, sigma, opts, and name-value arguments.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "d = eigs(Afun, n, ...)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eigs(A)",
        inputs: &INPUT_A,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eigs(A, k)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eigs(A, k, sigma)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eigs(A, k, sigma, opts)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eigs(A, k, sigma, Name, Value)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eigs(A, B, ...)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D] = eigs(...)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_VD,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D, flag] = eigs(...)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_VDFLAG,
    },
];

pub const EIGS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIGS.INVALID_ARGUMENT",
    identifier: Some("RunMat:eigs:InvalidArgument"),
    when: "Inputs, options, or requested output count are invalid.",
    message: "eigs: invalid argument",
};

pub const EIGS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIGS.INVALID_INPUT",
    identifier: Some("RunMat:eigs:InvalidInput"),
    when: "The matrix input is unsupported or has invalid shape.",
    message: "eigs: input matrix must be square",
};

pub const EIGS_ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIGS.UNSUPPORTED",
    identifier: Some("RunMat:eigs:UnsupportedInput"),
    when: "An unsupported operator storage or provider-only eigs form is requested.",
    message: "eigs: unsupported input form",
};

pub const EIGS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIGS.INTERNAL",
    identifier: Some("RunMat:eigs:Internal"),
    when: "Runtime cannot materialize selected eigen outputs.",
    message: "eigs: internal runtime failure",
};

pub const EIGS_ERROR_OPERATOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIGS.OPERATOR",
    identifier: Some("RunMat:eigs:Operator"),
    when: "A native coefficient application fails numerically or structurally.",
    message: "eigs: coefficient application failed",
};
pub const EIGS_ERROR_FACTORIZATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIGS.FACTORIZATION",
    identifier: Some("RunMat:eigs:Factorization"),
    when: "A reusable inverse factor is singular, unusable, or has invalid geometry.",
    message: "eigs: reusable inverse action failed",
};
pub const EIGS_ERROR_PROJECTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIGS.PROJECTED",
    identifier: Some("RunMat:eigs:ProjectedNoConvergence"),
    when: "The bounded projected eigendecomposition does not converge.",
    message: "eigs: projected eigendecomposition did not converge",
};
pub const EIGS_ERROR_BREAKDOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIGS.BREAKDOWN",
    identifier: Some("RunMat:eigs:NumericalBreakdown"),
    when: "The positive B-metric invariant subspace has fewer finite eigenpairs than requested.",
    message: "eigs: insufficient finite invariant subspace",
};
pub const EIGS_ERROR_DENSE_WORKSPACE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIGS.DENSE_WORKSPACE",
    identifier: Some("RunMat:eigs:DenseWorkspace"),
    when:
        "The exact dense route cannot capture or retain its original execution allocation authority; actual workspace and output admission errors retain their allocation causes.",
    message: "eigs: dense execution allocation authority is unavailable",
};
const ERRORS: [BuiltinErrorDescriptor; 9] = [
    EIGS_ERROR_INVALID_ARGUMENT,
    EIGS_ERROR_INVALID_INPUT,
    EIGS_ERROR_UNSUPPORTED,
    EIGS_ERROR_INTERNAL,
    EIGS_ERROR_OPERATOR,
    EIGS_ERROR_FACTORIZATION,
    EIGS_ERROR_PROJECTED,
    EIGS_ERROR_BREAKDOWN,
    EIGS_ERROR_DENSE_WORKSPACE,
];

pub const EIGS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
