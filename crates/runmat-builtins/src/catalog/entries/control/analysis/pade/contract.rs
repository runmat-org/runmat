use crate::*;

const DELAY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite nonnegative real Single or Double scalar delay in seconds.",
};
const ORDER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Positive platform integer order, supplied as a real floating or typed integer scalar.",
};
const NUM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "num",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numerator row in descending powers of s, matching T's floating class.",
};
const DEN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "den",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Monic denominator row in descending powers of s.",
};
const INPUTS: [BuiltinParamDescriptor; 2] = [DELAY, ORDER];
const OUTPUTS: [BuiltinParamDescriptor; 2] = [NUM, DEN];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[num,den] = pade(T,N)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const PADE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PADE.INVALID_ARGUMENT",
    identifier: Some("RunMat:pade:InvalidArgument"),
    when: "Delay or order is not a supported finite scalar, or the argument count is invalid.",
    message: "pade: invalid argument",
};
pub const PADE_ERROR_UNSUPPORTED_FORM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PADE.UNSUPPORTED_FORM",
    identifier: Some("RunMat:pade:UnsupportedForm"),
    when: "The call requests plotting or approximation of a dynamic model with stored delays.",
    message: "pade: unsupported form",
};
pub const PADE_ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PADE.NUMERICAL",
    identifier: Some("RunMat:pade:NumericalFailure"),
    when: "A normalized coefficient is not representable in the selected floating class.",
    message: "pade: numerical failure",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    PADE_ERROR_INVALID_ARGUMENT,
    PADE_ERROR_UNSUPPORTED_FORM,
    PADE_ERROR_NUMERICAL,
];

pub const PADE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_T: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "T",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed integer delays are not silently converted to floating seconds.",
};
const INTEGER_N: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "N",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed integer orders use the canonical exact platform-dimension decoder.",
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [INTEGER_T, INTEGER_N];
pub const PADE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[num,den] = pade(integer_T,integer_N)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed integer order N is supported; typed integer delay T is rejected.",
    }];
