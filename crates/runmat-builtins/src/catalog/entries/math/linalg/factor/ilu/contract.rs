use crate::*;

const A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square sparse single- or double-precision matrix.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("struct('type','nofill','droptol',0,'milu','off','udiag',0,'thresh',1)"),
    description: "Incomplete-LU options.",
};
const L: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse unit lower factor.",
};
const U: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "U",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse upper factor.",
};
const P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "P",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse permutation matrix.",
};
const W: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "W",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Combined sparse factors L + U - I.",
};
const IN1: [BuiltinParamDescriptor; 1] = [A];
const IN2: [BuiltinParamDescriptor; 2] = [A, OPTIONS];
const OUT1: [BuiltinParamDescriptor; 1] = [W];
const OUT2: [BuiltinParamDescriptor; 2] = [L, U];
const OUT3: [BuiltinParamDescriptor; 3] = [L, U, P];
pub const ILU_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "W = ilu(A)",
        inputs: &IN1,
        outputs: &OUT1,
    },
    BuiltinSignatureDescriptor {
        label: "[L,U] = ilu(A)",
        inputs: &IN1,
        outputs: &OUT2,
    },
    BuiltinSignatureDescriptor {
        label: "[L,U,P] = ilu(A)",
        inputs: &IN1,
        outputs: &OUT3,
    },
    BuiltinSignatureDescriptor {
        label: "W = ilu(A, options)",
        inputs: &IN2,
        outputs: &OUT1,
    },
    BuiltinSignatureDescriptor {
        label: "[L,U] = ilu(A, options)",
        inputs: &IN2,
        outputs: &OUT2,
    },
    BuiltinSignatureDescriptor {
        label: "[L,U,P] = ilu(A, options)",
        inputs: &IN2,
        outputs: &OUT3,
    },
];
pub const ILU_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ILU.INVALID_ARGUMENT",
    identifier: Some("RunMat:ilu:InvalidArgument"),
    when: "Options or output count are invalid.",
    message: "ilu: invalid options or output count",
};
pub const ILU_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ILU.INVALID_INPUT",
    identifier: Some("RunMat:ilu:InvalidInput"),
    when: "Input is not a finite square sparse floating matrix.",
    message: "ilu: expected a finite square sparse single- or double-precision matrix",
};
pub const ILU_ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ILU.NUMERICAL",
    identifier: Some("RunMat:ilu:NumericalFailure"),
    when: "An unreplaced zero pivot or numerical breakdown occurs.",
    message: "ilu: incomplete factorization encountered a numerical failure",
};
pub const ILU_ERROR_RUNTIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ILU.RUNTIME",
    identifier: Some("RunMat:ilu:Runtime"),
    when: "Managed execution or output publication fails.",
    message: "ilu: managed sparse execution failed",
};
const ERR: [BuiltinErrorDescriptor; 4] = [
    ILU_ERROR_INVALID_ARGUMENT,
    ILU_ERROR_INVALID_INPUT,
    ILU_ERROR_NUMERICAL,
    ILU_ERROR_RUNTIME,
];
pub const ILU_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ILU_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERR,
};
