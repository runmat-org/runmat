use crate::*;

const OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Coefficient estimates, one column per ridge parameter.",
}];

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Response vector with one value per observation.",
};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predictor matrix with observations in rows and predictors in columns.",
};

const PARAM_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonnegative ridge parameters.",
};

const PARAM_SCALED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "scaled",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Scaling flag: 1 returns standardized coefficients, 0 restores original scale and intercept.",
};

const INPUTS_Y_X_K: [BuiltinParamDescriptor; 3] = [PARAM_Y, PARAM_X, PARAM_K];
const INPUTS_Y_X_K_SCALED: [BuiltinParamDescriptor; 4] = [PARAM_Y, PARAM_X, PARAM_K, PARAM_SCALED];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = ridge(y, X, k)",
        inputs: &INPUTS_Y_X_K,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = ridge(y, X, k, scaled)",
        inputs: &INPUTS_Y_X_K_SCALED,
        outputs: &OUTPUT_B,
    },
];

pub const RIDGE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RIDGE.INVALID_ARGUMENT",
    identifier: Some("RunMat:ridge:InvalidArgument"),
    when: "Inputs, dimensions, ridge parameters, or scaling flag are malformed.",
    message: "ridge: invalid argument",
};

pub const RIDGE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RIDGE.INTERNAL",
    identifier: Some("RunMat:ridge:Internal"),
    when: "RunMat cannot allocate or construct ridge outputs.",
    message: "ridge: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [RIDGE_ERROR_INVALID_ARGUMENT, RIDGE_ERROR_INTERNAL];

pub const RIDGE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
