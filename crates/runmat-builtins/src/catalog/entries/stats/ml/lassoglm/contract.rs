use crate::*;

pub const LASSOGLM_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lassoglm-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "lassoglm with native-class integer predictor or response data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LassoglmIntegerDataExtension"),
    };
pub const LASSOGLM_INTEGER_PARAMETER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lassoglm-integer-parameter",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lassoglm with native-class integer floating parameters is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LassoglmIntegerParameterExtension"),
    };
pub const LASSOGLM_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lassoglm-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lassoglm with native-class integer structural controls is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LassoglmIntegerControlExtension"),
    };
pub const LASSOGLM_INTEGER_BOOLEAN_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lassoglm-integer-boolean",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lassoglm with native-class integer logical controls is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LassoglmIntegerBooleanExtension"),
    };
pub const LASSOGLM_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lassoglm-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lassoglm host fallback for explicit gpuArray inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LassoglmResidentInputExtension"),
    };
pub const LASSOGLM_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    LASSOGLM_INTEGER_DATA_EXTENSION,
    LASSOGLM_INTEGER_PARAMETER_EXTENSION,
    LASSOGLM_INTEGER_CONTROL_EXTENSION,
    LASSOGLM_INTEGER_BOOLEAN_EXTENSION,
    LASSOGLM_RESIDENT_INPUT_EXTENSION,
];

const INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "X or Y", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Native integer predictor or response data is gated before gather and checked at the binary64 solver boundary." }];
const INTEGER_PARAMETER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "Alpha, BinomialSize, Lambda, LambdaRatio, Offset, RelTol/TolX/TolFun, Weights, or Options.TolX/TolFun", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Native integer floating parameters are gated and require exact binary64 representation." }];
const INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "CV, MaxIter, MCReps, NumLambda, or Options.MaxIter",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Native integer structural controls are gated and decoded exactly.",
    }];
const INTEGER_BOOLEAN_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Constant/Intercept, Standardize, or Options.UseParallel",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Native integer logical controls accept only scalar zero or one, including scalar integer tensors.",
    }];
pub const LASSOGLM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "B = lassoglm(integer_X, integer_Y, distr, ___)", inputs: &INTEGER_DATA_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only integer data crosses a checked binary64 solver boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "B = lassoglm(X, Y, distr, Name, integer_value)", inputs: &INTEGER_PARAMETER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Floating parameters reject lossy integer conversion." },
    BuiltinIntegerCapabilityDescriptor { form: "B = lassoglm(X, Y, distr, Name, integer_control)", inputs: &INTEGER_CONTROL_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Structural controls are parsed exactly; ordinary integer-valued doubles remain documented." },
    BuiltinIntegerCapabilityDescriptor { form: "B = lassoglm(X, Y, distr, Name, integer_boolean)", inputs: &INTEGER_BOOLEAN_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Logical controls accept native integer scalar zero or one as a RunMat extension." },
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
        description:
            "Fit information structure containing Lambda, Intercept, Deviance, DF, and diagnostics.",
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
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Response vector, or two-column binomial successes/trials matrix.",
};

const PARAM_DISTR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "distr",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Distribution name: normal, binomial, or poisson.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as Lambda, Alpha, Standardize, Intercept, Weights, Offset, CV, NumLambda, LambdaRatio, MaxIter, RelTol, and Options.",
};

const INPUTS_REQUIRED: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_DISTR];
const INPUTS_FULL: [BuiltinParamDescriptor; 4] = [PARAM_X, PARAM_Y, PARAM_DISTR, PARAM_OPTIONS];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "B = lassoglm(X, Y, distr)",
        inputs: &INPUTS_REQUIRED,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = lassoglm(X, Y, distr, Name, Value)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "[B, FitInfo] = lassoglm(X, Y, distr)",
        inputs: &INPUTS_REQUIRED,
        outputs: &OUTPUT_B_FITINFO,
    },
    BuiltinSignatureDescriptor {
        label: "[B, FitInfo] = lassoglm(X, Y, distr, Name, Value)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUT_B_FITINFO,
    },
];

pub const LASSOGLM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LASSOGLM.INVALID_ARGUMENT",
    identifier: Some("RunMat:lassoglm:InvalidArgument"),
    when: "Inputs, distribution name, dimensions, option names, or option values are malformed.",
    message: "lassoglm: invalid argument",
};

pub const LASSOGLM_ERROR_CONVERGENCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LASSOGLM.CONVERGENCE",
    identifier: Some("RunMat:lassoglm:Convergence"),
    when: "The regularized GLM solver cannot make numerical progress.",
    message: "lassoglm: convergence failure",
};

pub const LASSOGLM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LASSOGLM.INTERNAL",
    identifier: Some("RunMat:lassoglm:Internal"),
    when: "RunMat cannot construct lassoglm outputs.",
    message: "lassoglm: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    LASSOGLM_ERROR_INVALID_ARGUMENT,
    LASSOGLM_ERROR_CONVERGENCE,
    LASSOGLM_ERROR_INTERNAL,
];

pub const LASSOGLM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
