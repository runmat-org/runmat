use crate::*;

pub const ICHOL_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse incomplete lower or upper Cholesky factor.",
}];

pub const ICHOL_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square sparse single- or double-precision matrix.",
}];

pub const ICHOL_INPUTS_A_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Square sparse single- or double-precision matrix.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: Some(
            "struct('type','nofill','droptol',0,'michol','off','diagcomp',0,'shape','lower')",
        ),
        description: "Incomplete-factorization options.",
    },
];

pub const ICHOL_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "L = ichol(A)",
        inputs: &ICHOL_INPUTS_A,
        outputs: &ICHOL_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "L = ichol(A, options)",
        inputs: &ICHOL_INPUTS_A_OPTIONS,
        outputs: &ICHOL_OUTPUTS,
    },
];

pub const ICHOL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ICHOL.INVALID_ARGUMENT",
    identifier: Some("RunMat:ichol:InvalidArgument"),
    when: "Options are malformed, unsupported, or more than one output is requested.",
    message: "ichol: invalid options or output count",
};
pub const ICHOL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ICHOL.INVALID_INPUT",
    identifier: Some("RunMat:ichol:InvalidInput"),
    when: "Input is not a finite square sparse single- or double-precision matrix.",
    message: "ichol: expected a finite square sparse single- or double-precision matrix",
};
pub const ICHOL_ERROR_NOT_POSITIVE_DEFINITE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ICHOL.NOT_POSITIVE_DEFINITE",
    identifier: Some("RunMat:ichol:NotPositiveDefinite"),
    when: "The incomplete factorization encounters a nonpositive pivot.",
    message: "ichol: matrix is not positive definite; use diagcomp to compensate the diagonal",
};
pub const ICHOL_ERROR_RUNTIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ICHOL.RUNTIME",
    identifier: Some("RunMat:ichol:Runtime"),
    when: "Managed execution or exact sparse output publication fails.",
    message: "ichol: managed sparse execution failed",
};
pub const ICHOL_ERRORS: [BuiltinErrorDescriptor; 4] = [
    ICHOL_ERROR_INVALID_ARGUMENT,
    ICHOL_ERROR_INVALID_INPUT,
    ICHOL_ERROR_NOT_POSITIVE_DEFINITE,
    ICHOL_ERROR_RUNTIME,
];

pub const ICHOL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ICHOL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ICHOL_ERRORS,
};
