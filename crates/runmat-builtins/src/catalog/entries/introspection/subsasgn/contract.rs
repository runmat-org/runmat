use crate::*;
const SUBSASGN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated object value.",
}];

const SUBSASGN_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        description: "Standard substruct-compatible assignment path.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];

const SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = subsasgn(obj, S, rhs)",
    inputs: &SUBSASGN_INPUTS,
    outputs: &SUBSASGN_OUTPUT,
}];

pub const SUBSASGN_ERROR_PATH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBSASGN.INVALID_PATH",
    identifier: Some("RunMat:InvalidObjectSubscriptPath"),
    when: "S is not a nonempty standard substruct-compatible path.",
    message: "invalid subscript path",
};

// Default assignment and class-overload failures retain their original identities.
pub const SUBSASGN_ERRORS: [BuiltinErrorDescriptor; 2] =
    [SUBSASGN_ERROR_OUTPUTS, SUBSASGN_ERROR_PATH];

pub const SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUBSASGN_ERRORS,
};

const SUBSASGN_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "payload subscripts",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer selectors may be carried in the indexing payload passed to an overloaded object method.",
    },
    BuiltinIntegerInputCapability {
        name: "rhs",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Assigned integer values are forwarded to the object's subsasgn implementation with their exact class, shape, and residency.",
    },
];
pub const SUBSASGN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "obj = subsasgn(obj, S_with_integer_subscripts, integer_rhs)",
        inputs: &SUBSASGN_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Default numeric assignment supports integer receivers, selectors and right-hand-side values under the shared assignment conversion contract. Class overloads receive authoritative inputs without dispatch-time numeric conversion and determine the returned value.",
    }];

pub const SUBSASGN_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBSASGN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:subsasgn:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "subsasgn: expected at most one output",
};
