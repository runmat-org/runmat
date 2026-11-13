use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Caller variable name, or empty character array when unnamed.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "argNumber",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based caller argument index.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "name = inputname(argNumber)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];
pub const INPUTNAME_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTNAME.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:NotEnoughInputs"),
    when: "No argument index is provided.",
    message: "inputname: not enough input arguments",
};
pub const INPUTNAME_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTNAME.TOO_MANY_INPUTS",
    identifier: Some("RunMat:TooManyInputs"),
    when: "More than one argument index is provided.",
    message: "inputname: too many input arguments",
};
pub const INPUTNAME_ERROR_ARGUMENT_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTNAME.ARGUMENT_INVALID",
    identifier: Some("RunMat:InputnameArgumentInvalid"),
    when: "The argument index is not a positive integer scalar.",
    message: "inputname: argument index must be a positive integer scalar",
};
pub const INPUTNAME_ERROR_ARGUMENT_EXCEEDS_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTNAME.ARGUMENT_EXCEEDS_INPUTS", identifier: Some("RunMat:InputnameArgumentExceedsInputs"),
    when: "The requested argument number exceeds the caller's input count or no function-input callsite exists.",
    message: "inputname: argument number exceeds number of function inputs",
};
pub const INPUTNAME_ERRORS: [BuiltinErrorDescriptor; 4] = [
    INPUTNAME_ERROR_NOT_ENOUGH_INPUTS,
    INPUTNAME_ERROR_TOO_MANY_INPUTS,
    INPUTNAME_ERROR_ARGUMENT_INVALID,
    INPUTNAME_ERROR_ARGUMENT_EXCEEDS_INPUTS,
];
pub const INPUTNAME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INPUTNAME_ERRORS,
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "argNumber", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "All eight integer classes are read exactly as a scalar positive index; logical is not admitted.",
}];
pub const INPUTNAME_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "name = inputname(integer_argNumber)", inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::ScalarOnly,
    notes: "Integer storage is validated without conversion to double; the result is a character row or 0x0 empty character array.",
}];
