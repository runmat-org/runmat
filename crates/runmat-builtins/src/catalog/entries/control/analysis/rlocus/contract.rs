use crate::*;

const RLOCUS_OUTPUT_R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Closed-loop pole locations as a branches-by-gains matrix.",
};
const RLOCUS_OUTPUT_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Gain vector used to compute the root locus.",
};
const RLOCUS_INPUT_SYS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf or ss model.",
};
const RLOCUS_INPUT_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Finite nonnegative gain vector.",
};
const RLOCUS_OUTPUTS_R: [BuiltinParamDescriptor; 1] = [RLOCUS_OUTPUT_R];
const RLOCUS_OUTPUTS_R_K: [BuiltinParamDescriptor; 2] = [RLOCUS_OUTPUT_R, RLOCUS_OUTPUT_K];
const RLOCUS_INPUTS_SYS: [BuiltinParamDescriptor; 1] = [RLOCUS_INPUT_SYS];
const RLOCUS_INPUTS_SYS_K: [BuiltinParamDescriptor; 2] = [RLOCUS_INPUT_SYS, RLOCUS_INPUT_K];
const RLOCUS_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "r = rlocus(sys)",
        inputs: &RLOCUS_INPUTS_SYS,
        outputs: &RLOCUS_OUTPUTS_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = rlocus(sys, k)",
        inputs: &RLOCUS_INPUTS_SYS_K,
        outputs: &RLOCUS_OUTPUTS_R,
    },
    BuiltinSignatureDescriptor {
        label: "[r,k] = rlocus(sys)",
        inputs: &RLOCUS_INPUTS_SYS,
        outputs: &RLOCUS_OUTPUTS_R_K,
    },
    BuiltinSignatureDescriptor {
        label: "[r,k] = rlocus(sys, k)",
        inputs: &RLOCUS_INPUTS_SYS_K,
        outputs: &RLOCUS_OUTPUTS_R_K,
    },
];
pub const RLOCUS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RLOCUS.INVALID_ARGUMENT",
    identifier: Some("RunMat:rlocus:InvalidArgument"),
    when: "Inputs do not match supported rlocus invocation forms.",
    message: "rlocus: invalid argument",
};
pub const RLOCUS_ERROR_INVALID_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RLOCUS.INVALID_MODEL",
    identifier: Some("RunMat:rlocus:InvalidModel"),
    when: "Input system is not a valid SISO tf or ss object.",
    message: "rlocus: invalid model",
};
pub const RLOCUS_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RLOCUS.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:rlocus:UnsupportedModel"),
    when: "Model form is not supported by the current implementation.",
    message: "rlocus: unsupported model",
};
pub const RLOCUS_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RLOCUS.PLOT_FAILED",
    identifier: Some("RunMat:rlocus:PlotFailed"),
    when: "Statement-form plotting failed for reasons other than known nonfatal setup conditions.",
    message: "rlocus: plotting failed",
};
pub const RLOCUS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RLOCUS.INTERNAL",
    identifier: Some("RunMat:rlocus:Internal"),
    when: "Root-locus computation or output construction failed.",
    message: "rlocus: internal error",
};
const RLOCUS_ERRORS: [BuiltinErrorDescriptor; 5] = [
    RLOCUS_ERROR_INVALID_ARGUMENT,
    RLOCUS_ERROR_INVALID_MODEL,
    RLOCUS_ERROR_UNSUPPORTED_MODEL,
    RLOCUS_ERROR_PLOT_FAILED,
    RLOCUS_ERROR_INTERNAL,
];
pub const RLOCUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RLOCUS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RLOCUS_ERRORS,
};
pub const RLOCUS_INTEGER_GAIN_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rlocus-integer-gain",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rlocus accepts a typed-integer gain vector as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RlocusIntegerGainExtension"),
};
pub const RLOCUS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [RLOCUS_INTEGER_GAIN_EXTENSION];
pub const RLOCUS_AND_MODEL_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    RLOCUS_INTEGER_GAIN_EXTENSION,
    super::super::super::models::INTEGER_COEFFICIENT_EXTENSION,
    super::super::super::models::LOGICAL_COEFFICIENT_EXTENSION,
    super::super::super::models::INTEGER_SAMPLE_TIME_EXTENSION,
    super::super::super::models::SS_INTEGER_MATRIX_EXTENSION,
    super::super::super::models::SS_INTEGER_SAMPLE_TIME_EXTENSION,
];
const RLOCUS_INTEGER_GAIN_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "k",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public gain surface does not document typed integer classes; RunMat mode admits them only through a checked binary64 root-locus boundary.",
    }];
pub const RLOCUS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "r = rlocus(sys, integer_k)",
        inputs: &RLOCUS_INTEGER_GAIN_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Gain values are gated and checked before any provider lookup, then intentionally enter double polynomial-root computation; returned gains are double.",
    }];
