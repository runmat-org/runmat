use crate::*;

const OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Least-squares coefficient estimates.",
}];

const OUTPUT_B_BINT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "bint",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Confidence intervals for coefficient estimates.",
    },
];

const OUTPUT_B_BINT_R: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "bint",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Confidence intervals for coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "r",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Regression residuals for complete observations.",
    },
];

const OUTPUT_B_BINT_R_RINT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "bint",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Confidence intervals for coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "r",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Regression residuals for complete observations.",
    },
    BuiltinParamDescriptor {
        name: "rint",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Confidence intervals for residuals.",
    },
];

const OUTPUT_B_BINT_R_RINT_STATS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "bint",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Confidence intervals for coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "r",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Regression residuals for complete observations.",
    },
    BuiltinParamDescriptor {
        name: "rint",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Confidence intervals for residuals.",
    },
    BuiltinParamDescriptor {
        name: "stats",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row vector [R2 F p errorVariance].",
    },
];

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Response vector.",
};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Design matrix with observations in rows and model terms in columns.",
};

const PARAM_ALPHA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "alpha",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0.05"),
    description: "Significance level for coefficient and residual intervals.",
};

const INPUTS_Y_X: [BuiltinParamDescriptor; 2] = [PARAM_Y, PARAM_X];
const INPUTS_Y_X_ALPHA: [BuiltinParamDescriptor; 3] = [PARAM_Y, PARAM_X, PARAM_ALPHA];

const SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "b = regress(y, X)",
        inputs: &INPUTS_Y_X,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "b = regress(y, X, alpha)",
        inputs: &INPUTS_Y_X_ALPHA,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint] = regress(y, X)",
        inputs: &INPUTS_Y_X,
        outputs: &OUTPUT_B_BINT,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint] = regress(y, X, alpha)",
        inputs: &INPUTS_Y_X_ALPHA,
        outputs: &OUTPUT_B_BINT,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint, r] = regress(y, X)",
        inputs: &INPUTS_Y_X,
        outputs: &OUTPUT_B_BINT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint, r] = regress(y, X, alpha)",
        inputs: &INPUTS_Y_X_ALPHA,
        outputs: &OUTPUT_B_BINT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint, r, rint] = regress(y, X)",
        inputs: &INPUTS_Y_X,
        outputs: &OUTPUT_B_BINT_R_RINT,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint, r, rint] = regress(y, X, alpha)",
        inputs: &INPUTS_Y_X_ALPHA,
        outputs: &OUTPUT_B_BINT_R_RINT,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint, r, rint, stats] = regress(y, X)",
        inputs: &INPUTS_Y_X,
        outputs: &OUTPUT_B_BINT_R_RINT_STATS,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint, r, rint, stats] = regress(y, X, alpha)",
        inputs: &INPUTS_Y_X_ALPHA,
        outputs: &OUTPUT_B_BINT_R_RINT_STATS,
    },
];

pub const REGRESS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGRESS.INVALID_ARGUMENT",
    identifier: Some("RunMat:regress:InvalidArgument"),
    when: "Inputs, dimensions, alpha, or requested output counts are malformed.",
    message: "regress: invalid argument",
};

pub const REGRESS_ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGRESS.NUMERICAL",
    identifier: Some("RunMat:regress:Numerical"),
    when: "The regression design cannot be solved numerically.",
    message: "regress: numerical failure",
};

pub const REGRESS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGRESS.INTERNAL",
    identifier: Some("RunMat:regress:Internal"),
    when: "RunMat cannot allocate or construct regression outputs.",
    message: "regress: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    REGRESS_ERROR_INVALID_ARGUMENT,
    REGRESS_ERROR_NUMERICAL,
    REGRESS_ERROR_INTERNAL,
];

pub const REGRESS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const REGRESS_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "regress-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "regress accepts typed-integer response and design data as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RegressIntegerDataExtension"),
};
pub const REGRESS_INTEGER_ALPHA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "regress-integer-alpha",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "regress accepts a typed-integer alpha as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RegressIntegerAlphaExtension"),
    };
pub const REGRESS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    REGRESS_INTEGER_DATA_EXTENSION,
    REGRESS_INTEGER_ALPHA_EXTENSION,
];
const REGRESS_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double response data; typed integers cross a checked binary64 regression boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double design data; typed integers cross the same checked boundary independently of y.",
    },
];
const REGRESS_INTEGER_ALPHA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "alpha",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double alpha; a typed integer is admitted only when exact in binary64 and within the open unit interval.",
    }];
pub const REGRESS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "regress(integer_y, integer_X [, alpha])",
        inputs: &REGRESS_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact authoritative integer values are checked before transparent gather and then intentionally enter the binary64 least-squares domain; outputs are double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "regress(y, X, integer_alpha)",
        inputs: &REGRESS_INTEGER_ALPHA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The typed alpha extension is gated and checked before provider access; the statistical computation and outputs are double.",
    },
];
