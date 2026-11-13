use crate::*;

const A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real n-by-n state matrix.",
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
    arity: BuiltinParamArity::Optional,
    default: Some("eye(size(B,2))"),
    description: "Symmetric nonsingular m-by-m input weight.",
};
const S: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("zeros(size(B))"),
    description: "n-by-m cross weight.",
};
const X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Stabilizing real double n-by-n CARE solution.",
};
const L: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "n-by-1 real or complex double closed-loop poles.",
};
const G: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "G",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real double m-by-n feedback gain.",
};
const IN3: [BuiltinParamDescriptor; 3] = [A, B, Q];
const IN4: [BuiltinParamDescriptor; 4] = [A, B, Q, R];
const IN5: [BuiltinParamDescriptor; 5] = [A, B, Q, R, S];
const OUT1: [BuiltinParamDescriptor; 1] = [X];
const OUT2: [BuiltinParamDescriptor; 2] = [X, L];
const OUT3: [BuiltinParamDescriptor; 3] = [X, L, G];
const SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "X = care(A,B,Q)",
        inputs: &IN3,
        outputs: &OUT1,
    },
    BuiltinSignatureDescriptor {
        label: "X = care(A,B,Q,R)",
        inputs: &IN4,
        outputs: &OUT1,
    },
    BuiltinSignatureDescriptor {
        label: "X = care(A,B,Q,R,S)",
        inputs: &IN5,
        outputs: &OUT1,
    },
    BuiltinSignatureDescriptor {
        label: "[X,L] = care(A,B,Q)",
        inputs: &IN3,
        outputs: &OUT2,
    },
    BuiltinSignatureDescriptor {
        label: "[X,L] = care(A,B,Q,R)",
        inputs: &IN4,
        outputs: &OUT2,
    },
    BuiltinSignatureDescriptor {
        label: "[X,L] = care(A,B,Q,R,S)",
        inputs: &IN5,
        outputs: &OUT2,
    },
    BuiltinSignatureDescriptor {
        label: "[X,L,G] = care(A,B,Q)",
        inputs: &IN3,
        outputs: &OUT3,
    },
    BuiltinSignatureDescriptor {
        label: "[X,L,G] = care(A,B,Q,R)",
        inputs: &IN4,
        outputs: &OUT3,
    },
    BuiltinSignatureDescriptor {
        label: "[X,L,G] = care(A,B,Q,R,S)",
        inputs: &IN5,
        outputs: &OUT3,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    BuiltinErrorDescriptor {
        code: "RM.CARE.INVALID_ARGUMENT",
        identifier: Some("RunMat:care:InvalidArgument"),
        when: "The standard form or requested output count is invalid.",
        message: "care: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.CARE.UNSUPPORTED_FORM",
        identifier: Some("RunMat:care:UnsupportedForm"),
        when: "A descriptor, factor, or diagnosis form is requested.",
        message: "care: unsupported form",
    },
    BuiltinErrorDescriptor {
        code: "RM.CARE.INVALID_DIMENSIONS",
        identifier: Some("RunMat:care:InvalidDimensions"),
        when: "System and weight dimensions are incompatible.",
        message: "care: invalid dimensions",
    },
    BuiltinErrorDescriptor {
        code: "RM.CARE.UNSUPPORTED_INPUT",
        identifier: Some("RunMat:care:UnsupportedInput"),
        when: "The matrices are not finite real host-compatible inputs.",
        message: "care: unsupported input",
    },
    BuiltinErrorDescriptor {
        code: "RM.CARE.SOLVER_FAILED",
        identifier: Some("RunMat:care:SolverFailed"),
        when: "No validated stabilizing CARE solution is available.",
        message: "care: Riccati solver failed",
    },
    BuiltinErrorDescriptor {
        code: "RM.CARE.INTERNAL",
        identifier: Some("RunMat:care:Internal"),
        when: "Output construction fails.",
        message: "care: internal error",
    },
];
pub const CARE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
