use crate::*;
const SUBSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Indexed value.",
}];

const SUBSREF_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexable receiver; default indexing or a class overload applies.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard substruct-compatible indexing path.",
    },
];

const SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[varargout] = subsref(obj, S)",
    inputs: &SUBSREF_INPUTS,
    outputs: &SUBSREF_OUTPUT,
}];

pub const SUBSREF_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBSREF.RECEIVER_INVALID",
    identifier: Some("RunMat:InvalidObjectDispatch"),
    when: "Receiver is not an object or handle object.",
    message: "subsref: requires object receiver",
};

pub const SUBSREF_ERROR_METHOD_MISSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBSREF.METHOD_MISSING",
    identifier: Some("RunMat:MissingSubsref"),
    when: "Target class does not implement subsref.",
    message: "subsref: class does not define subsref for indexing operation",
};

pub const SUBSREF_ERRORS: [BuiltinErrorDescriptor; 2] =
    [SUBSREF_ERROR_RECEIVER_INVALID, SUBSREF_ERROR_METHOD_MISSING];

pub const SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUBSREF_ERRORS,
};

const SUBSREF_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "payload subscripts",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer selectors may be carried in the indexing payload passed to an overloaded object method.",
    }];
pub const SUBSREF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "out = subsref(obj, S_with_integer_subscripts)",
        inputs: &SUBSREF_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Default numeric indexing supports integer receivers and selectors. Class overloads receive authoritative selector values without numeric conversion; their result class and overflow behavior are dynamic.",
    }];
