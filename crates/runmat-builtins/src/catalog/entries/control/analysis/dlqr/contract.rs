use crate::*;

const A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real n-by-n discrete state matrix.",
};
const B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real n-by-m input matrix.",
};
const Q: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Symmetric n-by-n state weight.",
};
const R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive-definite m-by-m input weight.",
};
const N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("zeros(size(B))"),
    description: "n-by-m cross weight.",
};
const K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "K",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real double m-by-n feedback gain.",
};
const S: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Stabilizing real double n-by-n DARE solution.",
};
const P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "P",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "n-by-1 real or complex double closed-loop poles.",
};
const IN4: [BuiltinParamDescriptor; 4] = [A, B, Q, R];
const IN5: [BuiltinParamDescriptor; 5] = [A, B, Q, R, N];
const OUT1: [BuiltinParamDescriptor; 1] = [K];
const OUT2: [BuiltinParamDescriptor; 2] = [K, S];
const OUT3: [BuiltinParamDescriptor; 3] = [K, S, P];
const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "K = dlqr(A,B,Q,R)",
        inputs: &IN4,
        outputs: &OUT1,
    },
    BuiltinSignatureDescriptor {
        label: "K = dlqr(A,B,Q,R,N)",
        inputs: &IN5,
        outputs: &OUT1,
    },
    BuiltinSignatureDescriptor {
        label: "[K,S] = dlqr(A,B,Q,R)",
        inputs: &IN4,
        outputs: &OUT2,
    },
    BuiltinSignatureDescriptor {
        label: "[K,S] = dlqr(A,B,Q,R,N)",
        inputs: &IN5,
        outputs: &OUT2,
    },
    BuiltinSignatureDescriptor {
        label: "[K,S,P] = dlqr(A,B,Q,R)",
        inputs: &IN4,
        outputs: &OUT3,
    },
    BuiltinSignatureDescriptor {
        label: "[K,S,P] = dlqr(A,B,Q,R,N)",
        inputs: &IN5,
        outputs: &OUT3,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    BuiltinErrorDescriptor {
        code: "RM.DLQR.INVALID_ARGUMENT",
        identifier: Some("RunMat:dlqr:InvalidArgument"),
        when: "The matrix form or requested output count is invalid.",
        message: "dlqr: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.DLQR.INVALID_DIMENSIONS",
        identifier: Some("RunMat:dlqr:InvalidDimensions"),
        when: "System and weight dimensions are incompatible.",
        message: "dlqr: invalid dimensions",
    },
    BuiltinErrorDescriptor {
        code: "RM.DLQR.UNSUPPORTED_INPUT",
        identifier: Some("RunMat:dlqr:UnsupportedInput"),
        when: "The matrices are not finite real host-compatible inputs.",
        message: "dlqr: unsupported input",
    },
    BuiltinErrorDescriptor {
        code: "RM.DLQR.SOLVER_FAILED",
        identifier: Some("RunMat:dlqr:SolverFailed"),
        when: "No validated stabilizing DARE solution is available.",
        message: "dlqr: Riccati solver failed",
    },
    BuiltinErrorDescriptor {
        code: "RM.DLQR.INTERNAL",
        identifier: Some("RunMat:dlqr:Internal"),
        when: "Output construction fails.",
        message: "dlqr: internal error",
    },
];
pub const DLQR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
