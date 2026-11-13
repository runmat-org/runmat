use crate::*;
const GET_P_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dependent property backing value.",
}];

const GET_P_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Object receiver.",
}];

const GET_P_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = get.p(obj)",
    inputs: &GET_P_INPUTS,
    outputs: &GET_P_OUTPUT,
}];

pub const GET_P_ERRORS: [BuiltinErrorDescriptor; 2] = [
    GET_P_ERROR_OUTPUTS,
    BuiltinErrorDescriptor {
        code: "RM.GET_P.RECEIVER_INVALID",
        identifier: Some("RunMat:GetPReceiverInvalid"),
        when: "Receiver is not an object value.",
        message: "get.p: requires object receiver",
    },
];

pub const GET_P_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GET_P_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &GET_P_ERRORS,
};

pub const GET_P_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GET_P.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:get.p:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "get.p: expected at most one output",
};
pub const GET_P_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
 form: "Nested integer backing values",
 inputs: &[BuiltinIntegerInputCapability { name: "object backing field or assigned value", classes: &ALL_INTEGER_CLASSES,
 availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
 notes: "Forwarded without numeric conversion, including nested resident integer values." }],
 computation_domain: BuiltinIntegerComputationDomain::Structural,
 output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
 overflow: BuiltinIntegerOverflowRule::NotApplicable,
 backend: BuiltinIntegerBackendRule::HostAndGpu,
 overload: BuiltinIntegerOverloadKind::FunctionSpecific,
 notes: "The stored value is preserved exactly. Missing backing fields read as double zero; receiver must be a value object.",
}];
