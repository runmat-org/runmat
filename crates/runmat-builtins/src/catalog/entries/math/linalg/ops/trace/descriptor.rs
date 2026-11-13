use crate::*;

pub const TRACE_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "trace-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "trace accepts native integer matrices",
    error_identifier: Some("RunMat:compatibility:TraceIntegerInputExtension"),
};
pub const TRACE_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "trace-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "trace accepts logical matrices",
    error_identifier: Some("RunMat:compatibility:TraceLogicalInputExtension"),
};
pub const TRACE_CHAR_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "trace-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "trace accepts character matrices",
    error_identifier: Some("RunMat:compatibility:TraceCharacterInputExtension"),
};
pub const TRACE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TRACE_INTEGER_INPUT_EXTENSION,
    TRACE_LOGICAL_INPUT_EXTENSION,
    TRACE_CHAR_INPUT_EXTENSION,
];

pub const TRACE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public trace input is single or double. Native integer matrices are independently gated and summed exactly before a checked double result boundary.",
    }];
pub const TRACE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "t = trace(integer_A)",
        inputs: &TRACE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Diagonal values are accumulated in an exact wide integer domain; the call rejects a result outside binary64's exact integer range instead of rounding it.",
    }];

pub const TRACE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Diagonal-sum trace result.",
}];

pub const TRACE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix-like value.",
}];

pub const TRACE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "t = trace(A)",
    inputs: &TRACE_INPUTS,
    outputs: &TRACE_OUTPUT,
}];

pub const TRACE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRACE.INVALID_INPUT",
    identifier: Some("RunMat:trace:InvalidInput"),
    when: "Input is unsupported or not matrix-shaped.",
    message: "trace: input must be 2-D",
};

pub const TRACE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRACE.INTERNAL",
    identifier: Some("RunMat:trace:Internal"),
    when: "Runtime cannot materialize or transport trace results.",
    message: "trace: internal runtime failure",
};

pub const TRACE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [TRACE_ERROR_INVALID_INPUT, TRACE_ERROR_INTERNAL];

pub const TRACE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRACE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TRACE_ERRORS,
};
