use crate::*;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First value to convert.",
    },
    BuiltinParamDescriptor {
        name: "rest",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional independent values.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Positional converted values.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[out1, ...] = convertContainedStringsToChars(value1, ...)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const CONVERT_CONTAINED_STRINGS_TO_CHARS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "value",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Top-level and nested integer values retain their class, shape, and provider owner.",
}];
pub const CONVERT_CONTAINED_STRINGS_TO_CHARS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "[out1, ...] = convertContainedStringsToChars(value1, ...)", inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::PreserveInput,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostAndGpu,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "String values convert recursively through cells and structs; integer values, including resident handles, remain exact without gather.",
}];
