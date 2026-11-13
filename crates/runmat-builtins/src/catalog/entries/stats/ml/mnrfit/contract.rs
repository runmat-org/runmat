use crate::*;

pub const INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mnrfit-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mnrfit with a typed-integer predictor matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MnrfitIntegerXExtension"),
};
pub const INTEGER_Y_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mnrfit-integer-y",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mnrfit with a typed-integer response array is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MnrfitIntegerYExtension"),
};
pub const INTEGER_PARAMETER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mnrfit-integer-parameter",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mnrfit with typed-integer floating options is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MnrfitIntegerParameterExtension"),
};
pub const INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mnrfit-integer-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mnrfit with typed-integer structural controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MnrfitIntegerControlExtension"),
};
pub const LOGICAL_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mnrfit-logical-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mnrfit with logical predictor data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MnrfitLogicalXExtension"),
};
pub const RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mnrfit-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mnrfit host fallback for explicit gpuArray inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MnrfitResidentInputExtension"),
};
pub const LEGACY_OPTIONS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mnrfit-legacy-options",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "mnrfit Constant, MaxIter, TolX, Options, and Display option aliases are RunMat extensions",
    error_identifier: Some("RunMat:compatibility:MnrfitLegacyOptionsExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    INTEGER_X_EXTENSION,
    INTEGER_Y_EXTENSION,
    INTEGER_PARAMETER_EXTENSION,
    INTEGER_CONTROL_EXTENSION,
    LOGICAL_X_EXTENSION,
    RESIDENT_INPUT_EXTENSION,
    LEGACY_OPTIONS_EXTENSION,
];

const INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer predictors are gated before gather and checked at the binary64 solver boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "Y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The current public Y type list omits integer classes; typed-integer categories or grouped counts are a checked RunMat extension.",
    },
];
const INTEGER_PARAMETER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Weights or Tolerance",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer floating solver parameters are independently gated and require exact binary64 representation.",
    }];
const INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "IterationLimit",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Current MATLAB documents single and double IterationLimit values; native integer controls are a gated structural extension.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = mnrfit(integer_X, integer_Y, ___)",
        inputs: &INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer data crosses one checked binary64 host-solver boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = mnrfit(X, Y, Name, integer_parameter)",
        inputs: &INTEGER_PARAMETER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Floating option values reject lossy integer conversion.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = mnrfit(X, Y, 'IterationLimit', integer_control)",
        inputs: &INTEGER_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Structural controls are parsed exactly; the MaxIter and Options spellings are separately gated RunMat extensions.",
    },
];

const OUTPUT_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Coefficient matrix with one column per non-reference response category.",
};

const OUTPUT_DEV: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dev",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Model deviance.",
};

const OUTPUT_STATS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "stats",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Fit statistics structure; its fields and shapes are described in the reference documentation.",
};

const INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predictor matrix with observations in rows.",
};

const INPUT_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Response class vector or grouped counts matrix with one column per class.",
};

const INPUT_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Name-value options such as Model, Link, Constant, Weights, IterationLimit, Tolerance, and Options.",
};

const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_Y];
const INPUTS_X_Y_OPTIONS: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_Y, INPUT_OPTIONS];
const OUTPUTS_B: [BuiltinParamDescriptor; 1] = [OUTPUT_B];
const OUTPUTS_B_DEV: [BuiltinParamDescriptor; 2] = [OUTPUT_B, OUTPUT_DEV];
const OUTPUTS_B_DEV_STATS: [BuiltinParamDescriptor; 3] = [OUTPUT_B, OUTPUT_DEV, OUTPUT_STATS];

const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "B = mnrfit(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUTS_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = mnrfit(X, Y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUTS_B,
    },
    BuiltinSignatureDescriptor {
        label: "[B, dev] = mnrfit(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUTS_B_DEV,
    },
    BuiltinSignatureDescriptor {
        label: "[B, dev] = mnrfit(X, Y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUTS_B_DEV,
    },
    BuiltinSignatureDescriptor {
        label: "[B, dev, stats] = mnrfit(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUTS_B_DEV_STATS,
    },
    BuiltinSignatureDescriptor {
        label: "[B, dev, stats] = mnrfit(X, Y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUTS_B_DEV_STATS,
    },
];

pub const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MNRFIT.INVALID_ARGUMENT",
    identifier: Some("RunMat:mnrfit:InvalidArgument"),
    when: "Inputs, response values, dimensions, or options are malformed.",
    message: "mnrfit: invalid argument",
};

pub const ERROR_CONVERGENCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MNRFIT.CONVERGENCE",
    identifier: Some("RunMat:mnrfit:Convergence"),
    when: "The multinomial logistic solver cannot make numerical progress.",
    message: "mnrfit: convergence failure",
};

pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MNRFIT.INTERNAL",
    identifier: Some("RunMat:mnrfit:Internal"),
    when: "RunMat cannot allocate or construct mnrfit outputs.",
    message: "mnrfit: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_CONVERGENCE, ERROR_INTERNAL];

pub const MNRFIT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
