use crate::*;
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host logical scalar class/category membership result.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value inspected without payload materialization.",
    },
    BuiltinParamDescriptor {
        name: "type_name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Class or abstract type designator as string scalar or character row.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isa(A, type_name)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const ISA_ERROR_TYPE_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISA.TYPE_NAME_INVALID",
    identifier: None,
    when: "Second argument is not a string scalar or row character vector.",
    message: "isa: TYPE must be a string scalar or character vector",
};
pub const ISA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ISA_ERROR_TYPE_NAME_INVALID],
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "A", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Host and automatic-resident integer classes expose exact dtype categories; explicit gpuArray reports its wrapper class." }];
pub const ISA_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "tf = isa(integer_A, type_name)", inputs: &INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Logical, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Class/category membership uses exact dtype metadata and returns a host logical scalar without numeric conversion or provider transfer." }];
