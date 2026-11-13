use crate::*;

pub const SQUEEZE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value with singleton dimensions removed (2-D minimum preserved).",
}];

pub const SQUEEZE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array/value.",
}];

pub const SQUEEZE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = squeeze(A)",
    inputs: &SQUEEZE_INPUTS,
    outputs: &SQUEEZE_OUTPUT,
}];

pub const SQUEEZE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQUEEZE.INVALID_INPUT",
    identifier: Some("RunMat:squeeze:InvalidInput"),
    when: "A has an unsupported type for squeeze.",
    message: "squeeze: unsupported input type",
};

pub const SQUEEZE_ERRORS: [BuiltinErrorDescriptor; 1] = [SQUEEZE_ERROR_INVALID_INPUT];

pub const SQUEEZE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SQUEEZE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SQUEEZE_ERRORS,
};

pub const SQUEEZE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "squeeze is class-agnostic structural array manipulation and preserves every integer element without numeric conversion.",
    }];

pub const SQUEEZE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "B = squeeze(integer_A)",
        inputs: &SQUEEZE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Only shape metadata changes. Host values retain authoritative storage and resident values retain the same provider buffer through reshape metadata.",
    }];
