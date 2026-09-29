use crate::*;

pub const IMPULSE_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "impulse-integer-numeric-role",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "impulse with typed-integer model metadata or time input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ImpulseIntegerExtension"),
};
pub const IMPULSE_LOGICAL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "impulse-logical-numeric-role",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "impulse with logical model metadata or time input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ImpulseLogicalExtension"),
};
pub const IMPULSE_SINGLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "impulse-single-numeric-role",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "impulse with single-precision model metadata or time input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ImpulseSingleExtension"),
};
pub const IMPULSE_RESIDENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "impulse-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "impulse with resident model metadata or time input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ImpulseResidentExtension"),
};
pub const IMPULSE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    IMPULSE_INTEGER_EXTENSION,
    IMPULSE_LOGICAL_EXTENSION,
    IMPULSE_SINGLE_EXTENSION,
    IMPULSE_RESIDENT_EXTENSION,
];

const IMPULSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "sys numeric metadata",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer coefficients and timing metadata are outside the documented dynamic-system/time surface, are gated before model parsing, and must be exactly representable at the host floating simulation boundary when the RunMat extension is enabled.",
    },
    BuiltinIntegerInputCapability {
        name: "t or tFinal",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public impulse reference specifies numeric time values and shapes but no typed-integer class support, so every typed-integer time control is consistently classified as a RunMat-only extension.",
    },
];
pub const IMPULSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[y, t] = impulse(sys, integer_time?)",
        inputs: &IMPULSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Non-double and resident numeric roles are independent RunMat extensions. Typed integers cross to binary64 only after exact-representability validation; this bounded SISO tf implementation simulates on the host and returns double response/time arrays.",
    }];

const IMPULSE_OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Impulse response samples (column vector).",
}];
const IMPULSE_OUTPUT_Y_T: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Impulse response samples (column vector).",
    },
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Time samples (column vector).",
    },
];
const IMPULSE_INPUTS_SYS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf model.",
}];
const IMPULSE_INPUTS_SYS_TIME: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "sys",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "SISO tf model.",
    },
    BuiltinParamDescriptor {
        name: "time",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Final time scalar or explicit time vector.",
    },
];
const IMPULSE_SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "impulse(sys)",
        inputs: &IMPULSE_INPUTS_SYS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "impulse(sys, tFinal)",
        inputs: &IMPULSE_INPUTS_SYS_TIME,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "impulse(sys, t)",
        inputs: &IMPULSE_INPUTS_SYS_TIME,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "y = impulse(sys)",
        inputs: &IMPULSE_INPUTS_SYS,
        outputs: &IMPULSE_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = impulse(sys, tFinal)",
        inputs: &IMPULSE_INPUTS_SYS_TIME,
        outputs: &IMPULSE_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = impulse(sys, t)",
        inputs: &IMPULSE_INPUTS_SYS_TIME,
        outputs: &IMPULSE_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[y,t] = impulse(sys)",
        inputs: &IMPULSE_INPUTS_SYS,
        outputs: &IMPULSE_OUTPUT_Y_T,
    },
    BuiltinSignatureDescriptor {
        label: "[y,t] = impulse(sys, tFinal)",
        inputs: &IMPULSE_INPUTS_SYS_TIME,
        outputs: &IMPULSE_OUTPUT_Y_T,
    },
    BuiltinSignatureDescriptor {
        label: "[y,t] = impulse(sys, t)",
        inputs: &IMPULSE_INPUTS_SYS_TIME,
        outputs: &IMPULSE_OUTPUT_Y_T,
    },
];
pub const IMPULSE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPULSE.INVALID_ARGUMENT",
    identifier: Some("RunMat:impulse:InvalidArgument"),
    when: "Inputs do not match supported impulse invocation forms.",
    message: "impulse: invalid argument",
};
pub const IMPULSE_ERROR_INVALID_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPULSE.INVALID_MODEL",
    identifier: Some("RunMat:impulse:InvalidModel"),
    when: "Input system is not a supported tf object with valid required metadata.",
    message: "impulse: invalid model",
};
pub const IMPULSE_ERROR_INVALID_TIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPULSE.INVALID_TIME",
    identifier: Some("RunMat:impulse:InvalidTime"),
    when: "Time argument is invalid for the model class or sampling mode.",
    message: "impulse: invalid time input",
};
pub const IMPULSE_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPULSE.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:impulse:UnsupportedModel"),
    when: "Model is well-formed but unsupported by the current impulse implementation.",
    message: "impulse: unsupported model",
};
pub const IMPULSE_ERROR_DISCRETE_LIMIT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPULSE.DISCRETE_LIMIT",
    identifier: Some("RunMat:impulse:DiscreteLimit"),
    when: "Discrete simulation would exceed platform or configured sample limits.",
    message: "impulse: discrete simulation limit exceeded",
};
pub const IMPULSE_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPULSE.PLOT_FAILED",
    identifier: Some("RunMat:impulse:PlotFailed"),
    when: "Statement-form plotting failed for reasons other than known nonfatal setup conditions.",
    message: "impulse: plotting failed",
};
pub const IMPULSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPULSE.INTERNAL",
    identifier: Some("RunMat:impulse:Internal"),
    when: "Internal response assembly failed.",
    message: "impulse: internal error",
};
const IMPULSE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    IMPULSE_ERROR_INVALID_ARGUMENT,
    IMPULSE_ERROR_INVALID_MODEL,
    IMPULSE_ERROR_INVALID_TIME,
    IMPULSE_ERROR_UNSUPPORTED_MODEL,
    IMPULSE_ERROR_DISCRETE_LIMIT,
    IMPULSE_ERROR_PLOT_FAILED,
    IMPULSE_ERROR_INTERNAL,
];
pub const IMPULSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMPULSE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMPULSE_ERRORS,
};
