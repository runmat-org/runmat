use crate::*;

pub const CHOL_OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Upper or lower triangular Cholesky factor.",
}];

pub const CHOL_OUTPUT_RP: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper or lower triangular Cholesky factor.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Failure flag (0 means positive definite).",
    },
];

pub const CHOL_OUTPUT_RPP: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper or lower triangular Cholesky factor.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Failure flag (0 means positive definite).",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sparse fill-reducing permutation matrix or vector.",
    },
];

pub const CHOL_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix to factorize.",
}];

pub const CHOL_INPUTS_A_TRIANGLE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to factorize.",
    },
    BuiltinParamDescriptor {
        name: "triangle",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"upper\""),
        description: "Triangle form (`\"upper\"` or `\"lower\"`).",
    },
];

pub const CHOL_INPUTS_S_OUTPUT_FORM: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sparse input matrix to factorize.",
    },
    BuiltinParamDescriptor {
        name: "outputForm",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"matrix\""),
        description: "Sparse permutation form (`\"matrix\"` or `\"vector\"`).",
    },
];

pub const CHOL_INPUTS_S_TRIANGLE_OUTPUT_FORM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sparse input matrix to factorize.",
    },
    BuiltinParamDescriptor {
        name: "triangle",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"upper\""),
        description: "Triangle form (`\"upper\"` or `\"lower\"`).",
    },
    BuiltinParamDescriptor {
        name: "outputForm",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"matrix\""),
        description: "Sparse permutation form (`\"matrix\"` or `\"vector\"`).",
    },
];

pub const CHOL_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "R = chol(A)",
        inputs: &CHOL_INPUTS_A,
        outputs: &CHOL_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = chol(A, triangle)",
        inputs: &CHOL_INPUTS_A_TRIANGLE,
        outputs: &CHOL_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, p] = chol(A)",
        inputs: &CHOL_INPUTS_A,
        outputs: &CHOL_OUTPUT_RP,
    },
    BuiltinSignatureDescriptor {
        label: "[R, p] = chol(A, triangle)",
        inputs: &CHOL_INPUTS_A_TRIANGLE,
        outputs: &CHOL_OUTPUT_RP,
    },
    BuiltinSignatureDescriptor {
        label: "[R, p, P] = chol(S)",
        inputs: &CHOL_INPUTS_A,
        outputs: &CHOL_OUTPUT_RPP,
    },
    BuiltinSignatureDescriptor {
        label: "[R, p, P] = chol(S, outputForm)",
        inputs: &CHOL_INPUTS_S_OUTPUT_FORM,
        outputs: &CHOL_OUTPUT_RPP,
    },
    BuiltinSignatureDescriptor {
        label: "[R, p, P] = chol(S, triangle, outputForm)",
        inputs: &CHOL_INPUTS_S_TRIANGLE_OUTPUT_FORM,
        outputs: &CHOL_OUTPUT_RPP,
    },
];

pub const CHOL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHOL.INVALID_ARGUMENT",
    identifier: Some("RunMat:chol:InvalidArgument"),
    when: "Option arguments or requested output count are invalid.",
    message: "chol supports a third permutation output only for sparse input",
};

pub const CHOL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHOL.INVALID_INPUT",
    identifier: Some("RunMat:chol:InvalidInput"),
    when: "Input is unsupported or not a square matrix.",
    message: "chol: expected a square numeric matrix",
};

pub const CHOL_ERROR_NOT_POSITIVE_DEFINITE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHOL.NOT_POSITIVE_DEFINITE",
    identifier: Some("RunMat:chol:NotPositiveDefinite"),
    when: "Single-output form receives a matrix that is not positive definite.",
    message: "Matrix must be positive definite.",
};

pub const CHOL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHOL.INTERNAL",
    identifier: Some("RunMat:chol:Internal"),
    when: "Runtime cannot materialize Cholesky outputs.",
    message: "chol: internal runtime failure",
};

pub const CHOL_ERRORS: [BuiltinErrorDescriptor; 4] = [
    CHOL_ERROR_INVALID_ARGUMENT,
    CHOL_ERROR_INVALID_INPUT,
    CHOL_ERROR_NOT_POSITIVE_DEFINITE,
    CHOL_ERROR_INTERNAL,
];

pub const CHOL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CHOL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CHOL_ERRORS,
};

pub const CHOL_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "chol-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "chol with a typed-integer input matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CholIntegerInputExtension"),
};
pub const CHOL_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "chol-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "chol with a logical input matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CholLogicalInputExtension"),
};

pub const CHOL_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [CHOL_INTEGER_INPUT_EXTENSION, CHOL_LOGICAL_INPUT_EXTENSION];

pub const CHOL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented matrix classes are single and double. RunMat mode admits all eight real integer classes after an exact binary64-boundary check.",
    }];

pub const CHOL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "R = chol(integer_A,...) and [R,p] = chol(integer_A,...)",
        inputs: &CHOL_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer matrices enter the binary64 Cholesky computation only after compatibility and exactness checks. Resident integer input bypasses the floating provider hook, gathers exactly, and restores a double factor to the owning provider when possible.",
    }];
