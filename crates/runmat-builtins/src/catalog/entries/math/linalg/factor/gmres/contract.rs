use crate::*;

const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Square native single/double dense or CSC matrix, or a repeatable n-vector operator.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Native single/double real or complex column vector matching A.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional restart, tol, maxit, M1, M2, and x0 positions.",
    },
];

const OUTPUTS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Approximate solution column vector in b's native class.",
    },
    BuiltinParamDescriptor {
        name: "flag",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "0 converged, 1 iteration limit, 2 numerical preconditioner failure, 3 stagnation, or 4 scalar breakdown. Callback language and allocation/authority failures remain errors.",
    },
    BuiltinParamDescriptor {
        name: "relres",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Relative norm of the left-preconditioned residual of the returned iterate.",
    },
    BuiltinParamDescriptor {
        name: "iter",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-by-two outer and inner iteration counts.",
    },
    BuiltinParamDescriptor {
        name: "resvec",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description:
            "Left-preconditioned residual norm history, beginning with the initial residual.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[x,flag,relres,iter,resvec] = gmres(A,b,restart,tol,maxit,M1,M2,x0)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.GMRES.INVALID_INPUT",
        identifier: Some("RunMat:gmres:InvalidInput"),
        when: "Input count, geometry, restart, tolerance, or iteration count is invalid.",
        message: "gmres: invalid input",
    },
    BuiltinErrorDescriptor {
        code: "RM.GMRES.UNSUPPORTED_INPUT",
        identifier: Some("RunMat:gmres:UnsupportedInput"),
        when: "A selected class, residency, or operator route is unavailable.",
        message: "gmres: unsupported input form",
    },
    BuiltinErrorDescriptor {
        code: "RM.GMRES.HOST_ALLOCATION",
        identifier: Some("RunMat:gmres:HostAllocation"),
        when: "Exact host scratch or retained output storage cannot be admitted.",
        message: "gmres: host admission failed",
    },
];

pub const GMRES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
