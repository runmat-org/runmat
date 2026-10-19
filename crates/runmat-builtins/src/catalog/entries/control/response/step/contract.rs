use crate::*;

const STEP_OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Step response samples (column vector).",
}];
const STEP_OUTPUT_Y_T: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Step response samples (column vector).",
    },
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Time samples (column vector).",
    },
];
const STEP_INPUTS_SYS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf model.",
}];
const STEP_INPUTS_SYS_TIME: [BuiltinParamDescriptor; 2] = [
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
const STEP_INPUTS_MULTI_PLOT: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "sys1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First SISO tf model.",
    },
    BuiltinParamDescriptor {
        name: "style1",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Plot style for the first model.",
    },
    BuiltinParamDescriptor {
        name: "sys2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second SISO tf model.",
    },
    BuiltinParamDescriptor {
        name: "style2",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Plot style for the second model.",
    },
    BuiltinParamDescriptor {
        name: "tFinal",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Shared final time for the plotted models.",
    },
];
const STEP_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "step(sys)",
        inputs: &STEP_INPUTS_SYS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "step(sys, tFinal)",
        inputs: &STEP_INPUTS_SYS_TIME,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "step(sys, t)",
        inputs: &STEP_INPUTS_SYS_TIME,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "step(sys1, style1, sys2, style2, tFinal)",
        inputs: &STEP_INPUTS_MULTI_PLOT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "y = step(sys)",
        inputs: &STEP_INPUTS_SYS,
        outputs: &STEP_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = step(sys, tFinal)",
        inputs: &STEP_INPUTS_SYS_TIME,
        outputs: &STEP_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = step(sys, t)",
        inputs: &STEP_INPUTS_SYS_TIME,
        outputs: &STEP_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[y,t] = step(sys)",
        inputs: &STEP_INPUTS_SYS,
        outputs: &STEP_OUTPUT_Y_T,
    },
    BuiltinSignatureDescriptor {
        label: "[y,t] = step(sys, tFinal)",
        inputs: &STEP_INPUTS_SYS_TIME,
        outputs: &STEP_OUTPUT_Y_T,
    },
    BuiltinSignatureDescriptor {
        label: "[y,t] = step(sys, t)",
        inputs: &STEP_INPUTS_SYS_TIME,
        outputs: &STEP_OUTPUT_Y_T,
    },
];
pub const STEP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEP.INVALID_ARGUMENT",
    identifier: Some("RunMat:step:InvalidArgument"),
    when: "Inputs do not match supported step invocation forms.",
    message: "step: invalid argument",
};
pub const STEP_ERROR_INVALID_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEP.INVALID_MODEL",
    identifier: Some("RunMat:step:InvalidModel"),
    when: "Input system is not a supported tf object with valid required properties.",
    message: "step: invalid model",
};
pub const STEP_ERROR_INVALID_TIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEP.INVALID_TIME",
    identifier: Some("RunMat:step:InvalidTime"),
    when: "Time argument is invalid for the model class or sampling mode.",
    message: "step: invalid time input",
};
pub const STEP_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEP.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:step:UnsupportedModel"),
    when: "Model is well-formed but unsupported by the current step implementation.",
    message: "step: unsupported model",
};
pub const STEP_ERROR_DISCRETE_LIMIT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEP.DISCRETE_LIMIT",
    identifier: Some("RunMat:step:DiscreteLimit"),
    when: "Discrete simulation would exceed platform or configured sample limits.",
    message: "step: discrete simulation limit exceeded",
};
pub const STEP_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEP.PLOT_FAILED",
    identifier: Some("RunMat:step:PlotFailed"),
    when: "Statement-form plotting failed for reasons other than known nonfatal setup conditions.",
    message: "step: plotting failed",
};
pub const STEP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEP.INTERNAL",
    identifier: Some("RunMat:step:Internal"),
    when: "Internal response assembly failed.",
    message: "step: internal error",
};
const STEP_ERRORS: [BuiltinErrorDescriptor; 7] = [
    STEP_ERROR_INVALID_ARGUMENT,
    STEP_ERROR_INVALID_MODEL,
    STEP_ERROR_INVALID_TIME,
    STEP_ERROR_UNSUPPORTED_MODEL,
    STEP_ERROR_DISCRETE_LIMIT,
    STEP_ERROR_PLOT_FAILED,
    STEP_ERROR_INTERNAL,
];
pub const STEP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STEP_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STEP_ERRORS,
};

pub const STEP_INTEGER_TIME_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "step-integer-time",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "step with a native typed-integer time input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:StepIntegerTimeExtension"),
};
pub const STEP_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [STEP_INTEGER_TIME_EXTENSION];
pub const STEP_AND_MODEL_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    STEP_INTEGER_TIME_EXTENSION,
    super::super::super::models::INTEGER_COEFFICIENT_EXTENSION,
    super::super::super::models::LOGICAL_COEFFICIENT_EXTENSION,
    super::super::super::models::INTEGER_SAMPLE_TIME_EXTENSION,
];
const STEP_INTEGER_TIME_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "t or tFinal",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents positive scalar, two-element, and vector time forms without publishing native integer storage classes. RunMat conservatively gates typed integers and requires exact binary64 representation.",
    }];
pub const STEP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[y,tOut] = step(sys, integer_t_or_tFinal)",
        inputs: &STEP_INTEGER_TIME_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Compatibility admission and exactness checks occur before provider access. Automatic residency may gather through the exact owner; the host simulator and time/output arrays use the model's binary64 computation domain.",
    }];
