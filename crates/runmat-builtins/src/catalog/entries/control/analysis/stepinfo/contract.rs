use crate::*;

const INFO: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar structure of step-response metrics.",
}];
const SYS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO transfer-function model.",
}];
const Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sampled step response.",
};
const T: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Time samples.",
};
const YFINAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "yfinal",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Steady-state value.",
};
const Y_INPUT: [BuiltinParamDescriptor; 1] = [Y];
const Y_T: [BuiltinParamDescriptor; 2] = [Y, T];
const Y_T_FINAL: [BuiltinParamDescriptor; 3] = [Y, T, YFINAL];
const WITH_THRESHOLD: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "inputs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Model or sampled-response positional inputs.",
    },
    BuiltinParamDescriptor {
        name: "SettlingTimeThreshold",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Supported settling-band option name.",
    },
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite threshold strictly between zero and one.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "info = stepinfo(sys)",
        inputs: &SYS,
        outputs: &INFO,
    },
    BuiltinSignatureDescriptor {
        label: "info = stepinfo(y)",
        inputs: &Y_INPUT,
        outputs: &INFO,
    },
    BuiltinSignatureDescriptor {
        label: "info = stepinfo(y,t)",
        inputs: &Y_T,
        outputs: &INFO,
    },
    BuiltinSignatureDescriptor {
        label: "info = stepinfo(y,t,yfinal)",
        inputs: &Y_T_FINAL,
        outputs: &INFO,
    },
    BuiltinSignatureDescriptor {
        label: "info = stepinfo(..., 'SettlingTimeThreshold', tol)",
        inputs: &WITH_THRESHOLD,
        outputs: &INFO,
    },
];
pub const STEPINFO_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEPINFO.INVALID_ARGUMENT",
    identifier: Some("RunMat:stepinfo:InvalidArgument"),
    when: "Input forms, options or requested output count are unsupported.",
    message: "stepinfo: invalid argument",
};
pub const STEPINFO_ERROR_INVALID_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEPINFO.INVALID_DATA",
    identifier: Some("RunMat:stepinfo:InvalidData"),
    when: "Response, time, final-value or threshold data is malformed.",
    message: "stepinfo: invalid response data",
};
pub const STEPINFO_ERROR_INVALID_SYSTEM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEPINFO.INVALID_SYSTEM",
    identifier: Some("RunMat:stepinfo:InvalidSystem"),
    when: "The model is not a supported scalar SISO tf object.",
    message: "stepinfo: invalid system",
};
pub const STEPINFO_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEPINFO.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:stepinfo:UnsupportedModel"),
    when: "The model is well formed but outside the supported response family.",
    message: "stepinfo: unsupported model",
};
pub const STEPINFO_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEPINFO.INTERNAL",
    identifier: Some("RunMat:stepinfo:Internal"),
    when: "Response simulation or metric assembly fails internally.",
    message: "stepinfo: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    STEPINFO_ERROR_INVALID_ARGUMENT,
    STEPINFO_ERROR_INVALID_DATA,
    STEPINFO_ERROR_INVALID_SYSTEM,
    STEPINFO_ERROR_UNSUPPORTED_MODEL,
    STEPINFO_ERROR_INTERNAL,
];
pub const STEPINFO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const STEPINFO_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "stepinfo-integer-sampled-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "Native typed-integer sampled response, time or final value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StepinfoIntegerSampledDataExtension"),
    };
pub const STEPINFO_INTEGER_THRESHOLD_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "stepinfo-integer-threshold",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Native typed-integer settling threshold is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StepinfoIntegerThresholdExtension"),
    };
pub const STEPINFO_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    STEPINFO_INTEGER_DATA_EXTENSION,
    STEPINFO_INTEGER_THRESHOLD_EXTENSION,
];
pub const STEPINFO_AND_MODEL_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    STEPINFO_INTEGER_DATA_EXTENSION,
    STEPINFO_INTEGER_THRESHOLD_EXTENSION,
    super::super::super::models::INTEGER_COEFFICIENT_EXTENSION,
    super::super::super::models::LOGICAL_COEFFICIENT_EXTENSION,
    super::super::super::models::INTEGER_SAMPLE_TIME_EXTENSION,
];
const INTEGER_DATA: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability { name: "y", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Every native integer response sample must convert exactly to binary64 before floating-point metrics." },
    BuiltinIntegerInputCapability { name: "t", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Integer time samples must be exactly representable and ordered." },
    BuiltinIntegerInputCapability { name: "yfinal", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Integer steady-state values must be exactly representable." },
];
const INTEGER_THRESHOLD: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "SettlingTimeThreshold",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "An exactly represented threshold must also lie strictly between zero and one.",
}];
pub const STEPINFO_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "info = stepinfo(integer_y, integer_t, integer_yfinal?)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Mode admission and exact conversion precede host response metrics.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "info = stepinfo(..., 'SettlingTimeThreshold', integer_threshold)",
        inputs: &INTEGER_THRESHOLD,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Threshold admission is independent of sampled-data integer admission.",
    },
];
