use crate::*;

const K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "K",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Optimal m-by-n real double feedback gain.",
};
const S: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Stabilizing n-by-n real double Riccati solution.",
};
const E: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "e",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "n-by-1 real or complex double closed-loop poles.",
};
const A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "n-by-n real state matrix.",
};
const B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "n-by-m real input matrix.",
};
const Q: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "n-by-n finite symmetric state weight.",
};
const R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "m-by-m finite symmetric positive-definite input weight.",
};
const N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("zeros(size(B))"),
    description: "n-by-m cross weight.",
};
const SYS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "State-space ss object.",
};
const OUT_K: [BuiltinParamDescriptor; 1] = [K];
const OUT_KS: [BuiltinParamDescriptor; 2] = [K, S];
const OUT_KSE: [BuiltinParamDescriptor; 3] = [K, S, E];
const IN_ABQR: [BuiltinParamDescriptor; 4] = [A, B, Q, R];
const IN_ABQRN: [BuiltinParamDescriptor; 5] = [A, B, Q, R, N];
const IN_SYS_QR: [BuiltinParamDescriptor; 3] = [SYS, Q, R];
const IN_SYS_QRN: [BuiltinParamDescriptor; 4] = [SYS, Q, R, N];
const SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "K = lqr(A, B, Q, R)",
        inputs: &IN_ABQR,
        outputs: &OUT_K,
    },
    BuiltinSignatureDescriptor {
        label: "K = lqr(A, B, Q, R, N)",
        inputs: &IN_ABQRN,
        outputs: &OUT_K,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S] = lqr(A, B, Q, R)",
        inputs: &IN_ABQR,
        outputs: &OUT_KS,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S] = lqr(A, B, Q, R, N)",
        inputs: &IN_ABQRN,
        outputs: &OUT_KS,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S, e] = lqr(A, B, Q, R)",
        inputs: &IN_ABQR,
        outputs: &OUT_KSE,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S, e] = lqr(A, B, Q, R, N)",
        inputs: &IN_ABQRN,
        outputs: &OUT_KSE,
    },
    BuiltinSignatureDescriptor {
        label: "K = lqr(sys, Q, R)",
        inputs: &IN_SYS_QR,
        outputs: &OUT_K,
    },
    BuiltinSignatureDescriptor {
        label: "K = lqr(sys, Q, R, N)",
        inputs: &IN_SYS_QRN,
        outputs: &OUT_K,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S] = lqr(sys, Q, R)",
        inputs: &IN_SYS_QR,
        outputs: &OUT_KS,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S] = lqr(sys, Q, R, N)",
        inputs: &IN_SYS_QRN,
        outputs: &OUT_KS,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S, e] = lqr(sys, Q, R)",
        inputs: &IN_SYS_QR,
        outputs: &OUT_KSE,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S, e] = lqr(sys, Q, R, N)",
        inputs: &IN_SYS_QRN,
        outputs: &OUT_KSE,
    },
];
pub const LQR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.INVALID_ARGUMENT",
    identifier: Some("RunMat:lqr:InvalidArgument"),
    when: "Call shape, requested output count, or optional arguments are invalid.",
    message: "lqr: invalid argument",
};
pub const LQR_ERROR_INVALID_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.INVALID_MODEL",
    identifier: Some("RunMat:lqr:InvalidModel"),
    when: "State-space metadata is malformed or unsupported.",
    message: "lqr: invalid model",
};
pub const LQR_ERROR_INVALID_DIMENSIONS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.INVALID_DIMENSIONS",
    identifier: Some("RunMat:lqr:InvalidDimensions"),
    when: "System and weighting matrix dimensions are incompatible.",
    message: "lqr: invalid dimensions",
};
pub const LQR_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:lqr:UnsupportedInput"),
    when: "Input contains unsupported types, complex values, or non-finite data.",
    message: "lqr: unsupported input",
};
pub const LQR_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:lqr:UnsupportedModel"),
    when: "Requested model form is outside state-space LQR.",
    message: "lqr: unsupported model",
};
pub const LQR_ERROR_SOLVER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.SOLVER_FAILED",
    identifier: Some("RunMat:lqr:SolverFailed"),
    when: "Riccati solver cannot produce a stabilizing finite solution.",
    message: "lqr: Riccati solver failed",
};
pub const LQR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.INTERNAL",
    identifier: Some("RunMat:lqr:Internal"),
    when: "Runtime fails while constructing outputs.",
    message: "lqr: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 7] = [
    LQR_ERROR_INVALID_ARGUMENT,
    LQR_ERROR_INVALID_MODEL,
    LQR_ERROR_INVALID_DIMENSIONS,
    LQR_ERROR_UNSUPPORTED_INPUT,
    LQR_ERROR_UNSUPPORTED_MODEL,
    LQR_ERROR_SOLVER_FAILED,
    LQR_ERROR_INTERNAL,
];
pub const LQR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const LQR_INTEGER_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lqr-integer-matrix-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Typed-integer LQR matrix operands are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LqrIntegerMatrixInputExtension"),
};
pub const LQR_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [LQR_INTEGER_MATRIX_EXTENSION];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 5] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed integer values enter the real double Riccati domain.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed integer values enter the real double Riccati domain.",
    },
    BuiltinIntegerInputCapability {
        name: "Q",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed integer values enter the real double Riccati domain.",
    },
    BuiltinIntegerInputCapability {
        name: "R",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed integer values enter the real double Riccati domain.",
    },
    BuiltinIntegerInputCapability {
        name: "N",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed integer values enter the real double Riccati domain.",
    },
];
pub const LQR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "[K,S,e] = lqr(A,B,Q,R,N) or lqr(sys,Q,R,N) with integer matrices", inputs: &INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "All eight classes are admitted in RunMat mode for matrix operands; conversion to f64 occurs at host Riccati input parsing and can round wide integers." }];
