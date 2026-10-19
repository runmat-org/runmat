use crate::*;

pub const SINGLE_SYSTEM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-single-system",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback(sys1) with implicit unity feedback is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackSingleSystemExtension"),
};
pub const SCALAR_FORWARD_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-scalar-forward-system",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with a scalar forward-path system is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackScalarForwardSystemExtension"),
};
pub const SCALAR_SYSTEMS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-scalar-scalar-systems",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with scalar forward and feedback-path systems is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackScalarScalarSystemsExtension"),
};
pub const INTEGER_GAIN_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-integer-scalar-gain",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with a typed-integer scalar system gain is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackIntegerScalarGainExtension"),
};
pub const INTEGER_SIGN_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-integer-sign",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with a typed-integer sign is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackIntegerSignExtension"),
};
pub const LOGICAL_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-logical-numeric-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with a logical numeric system or sign is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackLogicalNumericInputExtension"),
};
pub const RESIDENT_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-resident-numeric-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with a resident numeric system or sign is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackResidentNumericInputExtension"),
};
pub const SINGLE_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feedback-single-numeric-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "feedback with a native-single numeric system or sign is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeedbackSingleNumericInputExtension"),
};
pub const FEEDBACK_EXTENSIONS: [BuiltinExtensionDescriptor; 8] = [
    SINGLE_SYSTEM_EXTENSION,
    SCALAR_FORWARD_EXTENSION,
    SCALAR_SYSTEMS_EXTENSION,
    INTEGER_GAIN_EXTENSION,
    INTEGER_SIGN_EXTENSION,
    LOGICAL_NUMERIC_EXTENSION,
    RESIDENT_NUMERIC_EXTENSION,
    SINGLE_NUMERIC_EXTENSION,
];
pub const FEEDBACK_AND_MODEL_EXTENSIONS: [BuiltinExtensionDescriptor; 11] = [
    SINGLE_SYSTEM_EXTENSION,
    SCALAR_FORWARD_EXTENSION,
    SCALAR_SYSTEMS_EXTENSION,
    INTEGER_GAIN_EXTENSION,
    INTEGER_SIGN_EXTENSION,
    LOGICAL_NUMERIC_EXTENSION,
    RESIDENT_NUMERIC_EXTENSION,
    SINGLE_NUMERIC_EXTENSION,
    super::super::super::models::INTEGER_COEFFICIENT_EXTENSION,
    super::super::super::models::LOGICAL_COEFFICIENT_EXTENSION,
    super::super::super::models::INTEGER_SAMPLE_TIME_EXTENSION,
];

const INTEGER_GAIN_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "sys1", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Typed-integer scalar forward gain is independently gated and converted once to floating/complex tf coefficients." },
    BuiltinIntegerInputCapability { name: "sys2", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Typed-integer scalar feedback gain is independently gated and converted once to floating/complex tf coefficients." },
];
const INTEGER_SIGN_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "sign", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Typed-integer sign is independently gated and must be exactly -1 or +1; unsigned classes express only +1." }];
pub const FEEDBACK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "sys = feedback(integer_gain, sys2) or feedback(sys1, integer_gain)", inputs: &INTEGER_GAIN_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "All eight typed integer classes are RunMat-only scalar gains. The result is a host tf object; wide integers can round at the explicit binary64 coefficient boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "sys = feedback(sys1, sys2, integer_sign)", inputs: &INTEGER_SIGN_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "All eight integer classes are admitted as a RunMat-only sign control. Only -1 and +1 are valid; output is a host tf object." },
];

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Closed-loop SISO tf object.",
}];
const SYS1: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys1",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Forward-path SISO tf object or RunMat scalar gain.",
};
const SYS2: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys2",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Feedback-path SISO tf object or scalar gain.",
};
const SIGN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sign",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("-1"),
    description: "-1 for negative or +1 for positive feedback.",
};
const INPUT_ONE: [BuiltinParamDescriptor; 1] = [SYS1];
const INPUT_TWO: [BuiltinParamDescriptor; 2] = [SYS1, SYS2];
const INPUT_THREE: [BuiltinParamDescriptor; 3] = [SYS1, SYS2, SIGN];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "sys = feedback(sys1)",
        inputs: &INPUT_ONE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "sys = feedback(sys1, sys2)",
        inputs: &INPUT_TWO,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "sys = feedback(sys1, sys2, sign)",
        inputs: &INPUT_THREE,
        outputs: &OUTPUT,
    },
];
pub const FEEDBACK_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEEDBACK.INVALID_ARGUMENT",
    identifier: Some("RunMat:feedback:InvalidArgument"),
    when: "Inputs do not match supported invocation forms.",
    message: "feedback: invalid argument",
};
pub const FEEDBACK_ERROR_INVALID_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEEDBACK.INVALID_MODEL",
    identifier: Some("RunMat:feedback:InvalidModel"),
    when: "Inputs are not supported SISO transfer-function models or scalar gains.",
    message: "feedback: invalid model",
};
pub const FEEDBACK_ERROR_INVALID_SIGN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEEDBACK.INVALID_SIGN",
    identifier: Some("RunMat:feedback:InvalidSign"),
    when: "Feedback sign is not -1 or +1.",
    message: "feedback: sign must be -1 or +1",
};
pub const FEEDBACK_ERROR_INVALID_SAMPLE_TIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEEDBACK.INVALID_SAMPLE_TIME",
    identifier: Some("RunMat:feedback:InvalidSampleTime"),
    when: "A transfer-function model has an invalid sample time or variable domain.",
    message: "feedback: invalid sample time",
};
pub const FEEDBACK_ERROR_INVALID_VARIABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEEDBACK.INVALID_VARIABLE",
    identifier: Some("RunMat:feedback:InvalidVariable"),
    when: "A transfer-function model has an invalid variable for its sample time.",
    message: "feedback: invalid variable",
};
pub const FEEDBACK_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEEDBACK.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:feedback:UnsupportedModel"),
    when: "Model has unsupported delays or incompatible sample times.",
    message: "feedback: unsupported model",
};
pub const FEEDBACK_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEEDBACK.INTERNAL",
    identifier: Some("RunMat:feedback:Internal"),
    when: "Closed-loop assembly failed.",
    message: "feedback: internal error",
};
pub const FEEDBACK_ERRORS: [BuiltinErrorDescriptor; 7] = [
    FEEDBACK_ERROR_INVALID_ARGUMENT,
    FEEDBACK_ERROR_INVALID_MODEL,
    FEEDBACK_ERROR_INVALID_SIGN,
    FEEDBACK_ERROR_INVALID_SAMPLE_TIME,
    FEEDBACK_ERROR_INVALID_VARIABLE,
    FEEDBACK_ERROR_UNSUPPORTED_MODEL,
    FEEDBACK_ERROR_INTERNAL,
];
pub const FEEDBACK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FEEDBACK_ERRORS,
};
