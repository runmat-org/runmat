use crate::*;

const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Square dense or sparse single or double coefficient matrix, or callable mapping an n-by-1 vector to a matching vector.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column right-hand side matching the class and order of A.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional tol, maxit, M1, M2, and x0 positions. Preconditioners are selected callable inverse actions or reusable native-class dense/CSC matrix solves; general CSC factors require unrestricted host allocation.",
    },
];

const OUTPUTS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Approximate solution column vector.",
    },
    BuiltinParamDescriptor {
        name: "flag",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Convergence flag: 0 converged, 1 iteration limit, 2 unusable numerical preconditioner result, 3 stagnation, 4 breakdown. Language and execution-control failures remain errors.",
    },
    BuiltinParamDescriptor {
        name: "relres",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Relative residual of the returned iterate.",
    },
    BuiltinParamDescriptor {
        name: "iter",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Iteration at which the returned iterate was computed.",
    },
    BuiltinParamDescriptor {
        name: "resvec",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Residual norm history, including the initial residual.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[x,flag,relres,iter,resvec] = pcg(A,b,tol,maxit,M1,M2,x0)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.PCG.INVALID_INPUT",
        identifier: Some("RunMat:pcg:InvalidInput"),
        when: "Input count, dimensions, tolerance, iteration count, or sparse CSC topology are invalid.",
        message: "pcg: invalid input",
    },
    BuiltinErrorDescriptor {
        code: "RM.PCG.UNSUPPORTED_INPUT",
        identifier: Some("RunMat:pcg:UnsupportedInput"),
        when: "Coefficient, class combination, or preconditioner requires an unavailable route.",
        message: "pcg: unsupported input form",
    },
    BuiltinErrorDescriptor {
        code: "RM.PCG.HOST_ALLOCATION",
        identifier: Some("RunMat:pcg:HostAllocation"),
        when: "The selected host policy cannot admit exact scratch or retained result storage.",
        message: "pcg: host admission failed",
    },
];

pub const PCG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
