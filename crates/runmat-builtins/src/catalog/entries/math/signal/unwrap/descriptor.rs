use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Phase values with jumps corrected by multiples of 2*pi.",
}];
const INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "P",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Wrapped phase angles in radians.",
};
const TOLERANCE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tol",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("pi"),
    description: "Jump tolerance; values below pi are treated as pi. Use [] for the default.",
};
const DIMENSION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("first non-singleton dimension"),
    description: "Dimension along which to unwrap.",
};
const CORE: [BuiltinParamDescriptor; 1] = [INPUT];
const WITH_TOL: [BuiltinParamDescriptor; 2] = [INPUT, TOLERANCE];
const WITH_DIM: [BuiltinParamDescriptor; 3] = [INPUT, TOLERANCE, DIMENSION];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Q = unwrap(P)",
        inputs: &CORE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = unwrap(P, tol)",
        inputs: &WITH_TOL,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = unwrap(P, tol, dim)",
        inputs: &WITH_DIM,
        outputs: &OUTPUT,
    },
];

pub const UNWRAP_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNWRAP.ARG_COUNT",
    identifier: Some("RunMat:unwrap:ArgCount"),
    when: "More than three input arguments are supplied.",
    message: "unwrap: expected unwrap(P), unwrap(P, tol), or unwrap(P, tol, dim)",
};
pub const UNWRAP_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNWRAP.INVALID_INPUT",
    identifier: Some("RunMat:unwrap:InvalidInput"),
    when: "Input cannot be converted to a real numeric/logical phase array.",
    message: "unwrap: expected real numeric input",
};
pub const UNWRAP_ERROR_INVALID_TOLERANCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNWRAP.INVALID_TOLERANCE",
    identifier: Some("RunMat:unwrap:InvalidTolerance"),
    when: "Tolerance is not a finite, nonnegative scalar.",
    message: "unwrap: tolerance must be a finite nonnegative scalar",
};
pub const UNWRAP_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNWRAP.INVALID_DIMENSION",
    identifier: Some("RunMat:unwrap:InvalidDimension"),
    when: "Dimension is non-numeric, non-integer, or less than one.",
    message: "unwrap: dimension must be a positive integer scalar",
};
pub const UNWRAP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNWRAP.INTERNAL",
    identifier: Some("RunMat:unwrap:Internal"),
    when: "Internal gather, tensor conversion, or allocation fails.",
    message: "unwrap: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    UNWRAP_ERROR_ARG_COUNT,
    UNWRAP_ERROR_INVALID_INPUT,
    UNWRAP_ERROR_INVALID_TOLERANCE,
    UNWRAP_ERROR_INVALID_DIMENSION,
    UNWRAP_ERROR_INTERNAL,
];
pub const UNWRAP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
