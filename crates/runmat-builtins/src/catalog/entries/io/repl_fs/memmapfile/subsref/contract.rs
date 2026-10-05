use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host property value or a decoded selection from the opened mapped backing.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = memmapfile.subsref(m, S)",
    inputs: &[
        super::super::method_contract::RECEIVER,
        super::super::method_contract::SUBSCRIPT,
    ],
    outputs: &OUTPUT,
}];

pub const MEMMAPFILE_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        super::super::MEMMAPFILE_ERROR_TOO_MANY_OUTPUTS,
        super::super::method_contract::INVALID_SUBSCRIPT,
        super::super::method_contract::CANCELLED,
        super::super::method_contract::PROVIDER_ERROR,
        super::super::MEMMAPFILE_ERROR_INVALID_FORMAT,
        super::super::MEMMAPFILE_ERROR_READ_FAILED,
    ],
};

pub(super) const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "value = m.Data(indices) or m.Data(record).field(indices)",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Selected mapped integer bytes decode into their declared signed or unsigned class without a Double conversion.",
    }];
