use crate::*;
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf", ty: BuiltinParamType::LogicalArray, arity: BuiltinParamArity::Required,
    default: None, description: "Host logical validity mask for numeric handle arrays, or a scalar logical for other candidates.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Graphics receiver object, numeric handle alias, or another value to test.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = ishandle(h)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];
pub const ISHANDLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISHANDLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:ishandle:InvalidArgument"),
    when: "The input count is not exactly one.",
    message: "ishandle: invalid argument",
};
pub const ISHANDLE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISHANDLE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:ishandle:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "ishandle: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    ISHANDLE_ERROR_INVALID_ARGUMENT,
    ISHANDLE_ERROR_TOO_MANY_OUTPUTS,
];
pub const ISHANDLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const ISHANDLE_INTEGER_HANDLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ishandle-integer-handle-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ishandle numeric integer handle aliases are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IshandleIntegerHandleExtension"),
    };
pub const ISHANDLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ISHANDLE_INTEGER_HANDLE_EXTENSION];
const INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "h", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Host integer aliases are read from authoritative native storage and must convert exactly to the existing f64 graphics encoding. Disabled extensions and nonrepresentable aliases return false, not an extension error.",
}];
pub const ISHANDLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "tf = ishandle(integer_handle_alias)",
    inputs: &INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::Predicate,
    output_class: BuiltinIntegerOutputClassRule::Logical, overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
    notes: "Numeric tensors retain their full shape, including low rank and empty axes. Integer scalars return scalar logical. Resident values are metadata-only false masks regardless of element class; no gather or provider work occurs.",
}];
