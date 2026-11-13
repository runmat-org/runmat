use crate::*;

const OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Evaluated polynomial values with X's shape.",
}];
const OUTPUT_Y_DELTA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Evaluated polynomial values with X's shape.",
    },
    BuiltinParamDescriptor {
        name: "delta",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prediction interval values computed from S.",
    },
];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Descending-power coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric evaluation points, of any shape.",
    },
];
const INPUTS_S: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Descending-power coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric evaluation points, of any shape.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional polyfit statistics structure; required for delta.",
    },
];
const INPUTS_S_MU: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Descending-power coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric evaluation points, of any shape.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Polyfit statistics structure or [].",
    },
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Centering/scaling values; first two elements are mean and nonzero scale.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "y = polyval(p, x)",
        inputs: &INPUTS,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = polyval(p, x, S)",
        inputs: &INPUTS_S,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = polyval(p, x, S, mu)",
        inputs: &INPUTS_S_MU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[y, delta] = polyval(p, x, S)",
        inputs: &INPUTS_S,
        outputs: &OUTPUT_Y_DELTA,
    },
    BuiltinSignatureDescriptor {
        label: "[y, delta] = polyval(p, x, S, mu)",
        inputs: &INPUTS_S_MU,
        outputs: &OUTPUT_Y_DELTA,
    },
];
pub const POLYVAL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYVAL.INVALID_ARGUMENT",
    identifier: Some("RunMat:polyval:InvalidArgument"),
    when: "S/mu or requested-output combination is malformed.",
    message: "polyval: invalid argument",
};
pub const POLYVAL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYVAL.INVALID_INPUT",
    identifier: Some("RunMat:polyval:InvalidInput"),
    when: "Coefficients or evaluation points cannot be interpreted as numeric input.",
    message: "polyval: invalid input",
};
const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYVAL.INTERNAL",
    identifier: Some("RunMat:polyval:Internal"),
    when: "Runtime fails while building values, deltas, or provider fallbacks.",
    message: "polyval: internal runtime failure",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    POLYVAL_ERROR_INVALID_ARGUMENT,
    POLYVAL_ERROR_INVALID_INPUT,
    ERROR_INTERNAL,
];
pub const POLYVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const POLYVAL_INTEGER_COEFFICIENTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "polyval-integer-coefficients",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "polyval accepts typed-integer polynomial coefficients as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PolyvalIntegerCoefficientsExtension"),
    };
pub const POLYVAL_INTEGER_POINTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "polyval-integer-points",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "polyval accepts typed-integer query points as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PolyvalIntegerPointsExtension"),
    };
pub const POLYVAL_INTEGER_OPTIONS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "polyval-integer-fit-options",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "polyval accepts typed-integer S or mu data as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PolyvalIntegerFitOptionsExtension"),
    };
pub const POLYVAL_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    POLYVAL_INTEGER_COEFFICIENTS_EXTENSION,
    POLYVAL_INTEGER_POINTS_EXTENSION,
    POLYVAL_INTEGER_OPTIONS_EXTENSION,
];

const INTEGER_P: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "p",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer coefficients cross only after exact binary64 validation.",
}];
const INTEGER_X: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "x",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer points cross only after exact binary64 validation.",
}];
const INTEGER_OPTIONS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "S or mu numeric fields",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer fit metadata is recursively checked before prediction-interval computation.",
}];
pub const POLYVAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = polyval(integer_p,x,___)",
        inputs: &INTEGER_P,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Coefficient class is gated before provider or host Horner evaluation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = polyval(p,integer_x,___)",
        inputs: &INTEGER_X,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Point class is gated before provider or host Horner evaluation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[y,delta] = polyval(p,x,integer_S_or_mu)",
        inputs: &INTEGER_OPTIONS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Fit metadata is independently gated before prediction-interval computation.",
    },
];
