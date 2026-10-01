use crate::*;

const OUTPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Least-squares coefficient estimates.",
}];

const OUTPUT_X_STDX: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "stdx",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard errors for the coefficient estimates.",
    },
];

const OUTPUT_X_STDX_MSE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "stdx",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard errors for the coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "mse",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mean squared weighted residual error for each response.",
    },
];

const OUTPUT_X_STDX_MSE_S: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "stdx",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard errors for the coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "mse",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mean squared weighted residual error.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Estimated covariance matrix for the coefficient estimates.",
    },
];

const PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Design matrix with observations in rows.",
};

const PARAM_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation vector or matrix with one row per observation.",
};

const PARAM_V: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "V",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("eye(size(A,1))"),
    description: "Observation covariance matrix or vector of observation weights.",
};

const PARAM_ALG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "alg",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("chol"),
    description: "Algorithm selector, either \"orth\" or \"chol\".",
};

const INPUTS_A_B: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_B];
const INPUTS_A_B_V: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_B, PARAM_V];
const INPUTS_A_B_ALG: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_B, PARAM_ALG];
const INPUTS_A_B_V_ALG: [BuiltinParamDescriptor; 4] = [PARAM_A, PARAM_B, PARAM_V, PARAM_ALG];

const SIGNATURES: [BuiltinSignatureDescriptor; 16] = [
    BuiltinSignatureDescriptor {
        label: "x = lscov(A, B)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = lscov(A, B, V)",
        inputs: &INPUTS_A_B_V,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = lscov(A, B, alg)",
        inputs: &INPUTS_A_B_ALG,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = lscov(A, B, V, alg)",
        inputs: &INPUTS_A_B_V_ALG,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx] = lscov(A, B)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUT_X_STDX,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx] = lscov(A, B, V)",
        inputs: &INPUTS_A_B_V,
        outputs: &OUTPUT_X_STDX,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx] = lscov(A, B, alg)",
        inputs: &INPUTS_A_B_ALG,
        outputs: &OUTPUT_X_STDX,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx] = lscov(A, B, V, alg)",
        inputs: &INPUTS_A_B_V_ALG,
        outputs: &OUTPUT_X_STDX,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse] = lscov(A, B)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUT_X_STDX_MSE,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse] = lscov(A, B, V)",
        inputs: &INPUTS_A_B_V,
        outputs: &OUTPUT_X_STDX_MSE,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse] = lscov(A, B, alg)",
        inputs: &INPUTS_A_B_ALG,
        outputs: &OUTPUT_X_STDX_MSE,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse] = lscov(A, B, V, alg)",
        inputs: &INPUTS_A_B_V_ALG,
        outputs: &OUTPUT_X_STDX_MSE,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse, S] = lscov(A, b)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUT_X_STDX_MSE_S,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse, S] = lscov(A, b, V)",
        inputs: &INPUTS_A_B_V,
        outputs: &OUTPUT_X_STDX_MSE_S,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse, S] = lscov(A, b, alg)",
        inputs: &INPUTS_A_B_ALG,
        outputs: &OUTPUT_X_STDX_MSE_S,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse, S] = lscov(A, b, V, alg)",
        inputs: &INPUTS_A_B_V_ALG,
        outputs: &OUTPUT_X_STDX_MSE_S,
    },
];

pub const LSCOV_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSCOV.INVALID_ARGUMENT",
    identifier: Some("RunMat:lscov:InvalidArgument"),
    when: "Inputs, dimensions, weighting arguments, algorithm, or requested output count are malformed.",
    message: "lscov: invalid argument",
};

pub const LSCOV_ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSCOV.NUMERICAL",
    identifier: Some("RunMat:lscov:Numerical"),
    when: "The weighted least-squares system cannot be solved numerically.",
    message: "lscov: numerical failure",
};

pub const LSCOV_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSCOV.INTERNAL",
    identifier: Some("RunMat:lscov:Internal"),
    when: "RunMat cannot allocate or construct lscov outputs.",
    message: "lscov: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    LSCOV_ERROR_INVALID_ARGUMENT,
    LSCOV_ERROR_NUMERICAL,
    LSCOV_ERROR_INTERNAL,
];

pub const LSCOV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
