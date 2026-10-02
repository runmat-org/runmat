use crate::*;

const OUTPUT_P: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Descending-power polynomial coefficient row vector.",
}];
const OUTPUT_PS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Descending-power polynomial coefficient row vector.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Fit statistics with R, df, and normr fields.",
    },
];
const OUTPUT_PSMU: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Descending-power polynomial coefficient row vector.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Fit statistics with R, df, and normr fields.",
    },
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Centering and scaling vector [mean(x), std(x)].",
    },
];
const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real sample locations as a vector.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real or complex fitted observations as a vector.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonnegative polynomial degree.",
    },
];
const INPUTS_WEIGHTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real sample locations as a vector.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real or complex fitted observations as a vector.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonnegative polynomial degree.",
    },
    BuiltinParamDescriptor {
        name: "weights",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Finite nonnegative weights, one per sample, with matching element count.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "p = polyfit(X, Y, n)",
        inputs: &INPUTS,
        outputs: &OUTPUT_P,
    },
    BuiltinSignatureDescriptor {
        label: "p = polyfit(X, Y, n, weights)",
        inputs: &INPUTS_WEIGHTS,
        outputs: &OUTPUT_P,
    },
    BuiltinSignatureDescriptor {
        label: "[p, S] = polyfit(X, Y, n)",
        inputs: &INPUTS,
        outputs: &OUTPUT_PS,
    },
    BuiltinSignatureDescriptor {
        label: "[p, S] = polyfit(X, Y, n, weights)",
        inputs: &INPUTS_WEIGHTS,
        outputs: &OUTPUT_PS,
    },
    BuiltinSignatureDescriptor {
        label: "[p, S, mu] = polyfit(X, Y, n)",
        inputs: &INPUTS,
        outputs: &OUTPUT_PSMU,
    },
    BuiltinSignatureDescriptor {
        label: "[p, S, mu] = polyfit(X, Y, n, weights)",
        inputs: &INPUTS_WEIGHTS,
        outputs: &OUTPUT_PSMU,
    },
];

pub const POLYFIT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYFIT.INVALID_ARGUMENT",
    identifier: Some("RunMat:polyfit:InvalidArgument"),
    when: "Degree or weight arguments are malformed or unsupported.",
    message: "polyfit: invalid argument",
};
pub const POLYFIT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYFIT.INVALID_INPUT",
    identifier: Some("RunMat:polyfit:InvalidInput"),
    when: "Input vectors or values cannot be processed for polynomial fitting.",
    message: "polyfit: invalid input",
};
pub const POLYFIT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYFIT.INTERNAL",
    identifier: Some("RunMat:polyfit:Internal"),
    when: "Runtime fails while constructing fit outputs or provider payloads.",
    message: "polyfit: internal runtime failure",
};
pub const POLYFIT_ERROR_PROVIDER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYFIT.PROVIDER_FAILED",
    identifier: Some("RunMat:polyfit:ProviderFailed"),
    when: "An owning acceleration provider returns malformed or nonfinite fit outputs.",
    message: "polyfit: invalid provider result",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    POLYFIT_ERROR_INVALID_ARGUMENT,
    POLYFIT_ERROR_INVALID_INPUT,
    POLYFIT_ERROR_INTERNAL,
    POLYFIT_ERROR_PROVIDER_FAILED,
];
pub const POLYFIT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const POLYFIT_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "polyfit-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "polyfit accepts typed-integer sample locations as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PolyfitIntegerXExtension"),
};
pub const POLYFIT_INTEGER_Y_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "polyfit-integer-y",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "polyfit accepts typed-integer sample values as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PolyfitIntegerYExtension"),
};
pub const POLYFIT_INTEGER_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "polyfit-integer-weights",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "polyfit accepts typed-integer observation weights as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PolyfitIntegerWeightsExtension"),
    };
pub const POLYFIT_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    POLYFIT_INTEGER_X_EXTENSION,
    POLYFIT_INTEGER_Y_EXTENSION,
    POLYFIT_INTEGER_WEIGHTS_EXTENSION,
];

const X_INTEGER: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed sample locations require exact binary64 conversion.",
}];
const Y_INTEGER: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Y",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed fitted values require exact binary64 conversion.",
}];
const DEGREE_INTEGER: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "n",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The nonnegative integer degree is decoded exactly as a structural width.",
}];
const WEIGHTS_INTEGER: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "weights",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed weights require exact binary64 conversion before weighted fitting.",
}];
pub const POLYFIT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "p = polyfit(integer_X,Y,n,___)",
        inputs: &X_INTEGER,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "X is independently gated before provider dispatch and QR fitting.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "p = polyfit(X,integer_Y,n,___)",
        inputs: &Y_INTEGER,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Y is independently gated before provider dispatch and QR fitting.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "p = polyfit(X,Y,integer_n,___)",
        inputs: &DEGREE_INTEGER,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Degree selects Vandermonde width without entering numerical fitting data.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "p = polyfit(X,Y,n,integer_weights)",
        inputs: &WEIGHTS_INTEGER,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed weights are independently gated before weighted least squares.",
    },
];
