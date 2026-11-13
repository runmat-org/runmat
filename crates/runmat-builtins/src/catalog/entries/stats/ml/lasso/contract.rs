use crate::*;

pub const LASSO_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lasso-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lasso with native-class integer predictor or response data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LassoIntegerDataExtension"),
};
pub const LASSO_INTEGER_PARAMETER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lasso-integer-parameter",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lasso with native-class integer floating parameters is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LassoIntegerParameterExtension"),
    };
pub const LASSO_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lasso-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lasso with native-class integer structural controls is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LassoIntegerControlExtension"),
    };
pub const LASSO_INTEGER_BOOLEAN_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lasso-integer-boolean",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lasso with native-class integer logical controls is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LassoIntegerBooleanExtension"),
    };
pub const LASSO_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lasso-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lasso host fallback for explicit gpuArray inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LassoResidentInputExtension"),
};

pub const LASSO_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    LASSO_INTEGER_DATA_EXTENSION,
    LASSO_INTEGER_PARAMETER_EXTENSION,
    LASSO_INTEGER_CONTROL_EXTENSION,
    LASSO_INTEGER_BOOLEAN_EXTENSION,
    LASSO_RESIDENT_INPUT_EXTENSION,
];

const INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X or y",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Native integer data is gated before gather and must be exactly representable at the binary64 solver boundary.",
}];
const INTEGER_PARAMETER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Alpha, Lambda, LambdaRatio, Weights, or RelTol",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Native integer floating parameters are gated and checked for exact binary64 representation.",
    }];
const INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "CV, DFmax, MaxIter, MCReps, or NumLambda",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Native integer structural controls are gated and decoded exactly without a floating round trip.",
    }];
const INTEGER_BOOLEAN_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Intercept, Standardize, or UseCovariance",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Native integer logical controls accept only scalar zero or one, including scalar integer tensors.",
    }];
pub const LASSO_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "B = lasso(integer_X, integer_y, ___)", inputs: &INTEGER_DATA_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only integer data crosses a checked binary64 solver boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "B = lasso(X, y, Name, integer_value)", inputs: &INTEGER_PARAMETER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Floating parameters reject lossy integer conversion." },
    BuiltinIntegerCapabilityDescriptor { form: "B = lasso(X, y, Name, integer_control)", inputs: &INTEGER_CONTROL_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Structural controls are parsed exactly; ordinary integer-valued doubles remain documented." },
    BuiltinIntegerCapabilityDescriptor { form: "B = lasso(X, y, Name, integer_boolean)", inputs: &INTEGER_BOOLEAN_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Logical controls accept native integer scalar zero or one as a RunMat extension." },
];

const OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Coefficient matrix with one column per Lambda value.",
}];

const OUTPUT_B_FITINFO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Coefficient matrix with one column per Lambda value.",
    },
    BuiltinParamDescriptor {
        name: "FitInfo",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Fit information structure containing Lambda, Alpha, Intercept, DF, MSE, and related fields.",
    },
];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predictor matrix with observations in rows and predictors in columns.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Response vector with one value per row of X.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as Lambda, Alpha, Standardize, Intercept, Weights, CV, NumLambda, LambdaRatio, MaxIter, and RelTol.",
};

const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUTS_X_Y_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_OPTIONS];

const LASSO_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "B = lasso(X, y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = lasso(X, y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "[B, FitInfo] = lasso(X, y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_B_FITINFO,
    },
    BuiltinSignatureDescriptor {
        label: "[B, FitInfo] = lasso(X, y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_B_FITINFO,
    },
];

pub const LASSO_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LASSO.INVALID_ARGUMENT",
    identifier: Some("RunMat:lasso:InvalidArgument"),
    when: "Inputs, option names, option values, dimensions, or tolerances are malformed.",
    message: "lasso: invalid argument",
};

pub const LASSO_ERROR_CONVERGENCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LASSO.CONVERGENCE",
    identifier: Some("RunMat:lasso:Convergence"),
    when: "Coordinate descent cannot make numerical progress for the supplied data.",
    message: "lasso: convergence failure",
};

pub const LASSO_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LASSO.INTERNAL",
    identifier: Some("RunMat:lasso:Internal"),
    when: "RunMat cannot construct lasso outputs.",
    message: "lasso: internal error",
};

const LASSO_ERRORS: [BuiltinErrorDescriptor; 3] = [
    LASSO_ERROR_INVALID_ARGUMENT,
    LASSO_ERROR_CONVERGENCE,
    LASSO_ERROR_INTERNAL,
];

pub const LASSO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LASSO_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LASSO_ERRORS,
};
