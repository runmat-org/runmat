use crate::*;
const SET_P_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated object receiver.",
}];

const SET_P_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object receiver.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value to assign.",
    },
];

const SET_P_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = set.p(obj, value)",
    inputs: &SET_P_INPUTS,
    outputs: &SET_P_OUTPUT,
}];

pub const SET_P_ERRORS: [BuiltinErrorDescriptor; 2] = [
    SET_P_ERROR_OUTPUTS,
    BuiltinErrorDescriptor {
        code: "RM.SET_P.RECEIVER_INVALID",
        identifier: Some("RunMat:SetPReceiverInvalid"),
        when: "Receiver is not an object value.",
        message: "set.p: requires object receiver",
    },
];

pub const SET_P_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SET_P_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &SET_P_ERRORS,
};

pub const SET_P_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SET_P.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:set.p:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "set.p: expected at most one output",
};
pub const SET_P_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
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
