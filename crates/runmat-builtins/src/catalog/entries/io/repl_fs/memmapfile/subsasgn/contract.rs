use crate::*;

const RHS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "rhs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value matching the selected Format class, shape, and record fields.",
};

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "m",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated memmapfile receiver; opened backing writes are observable by aliases.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "m = memmapfile.subsasgn(m, S, rhs)",
    inputs: &[
        super::super::method_contract::RECEIVER,
        super::super::method_contract::SUBSCRIPT,
        RHS,
    ],
    outputs: &OUTPUT,
}];

pub const MEMMAPFILE_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        super::super::MEMMAPFILE_ERROR_TOO_MANY_OUTPUTS,
        super::super::method_contract::INVALID_SUBSCRIPT,
        super::super::method_contract::CANCELLED,
        super::super::method_contract::PROVIDER_ERROR,
        super::super::MEMMAPFILE_ERROR_INVALID_CONTROL,
        super::super::MEMMAPFILE_ERROR_INVALID_FORMAT,
        super::super::MEMMAPFILE_ERROR_READ_FAILED,
        super::super::MEMMAPFILE_ERROR_WRITE_FAILED,
    ],
};

const INTEGER_RHS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "rhs",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "An integer mapped field accepts its own exact declared integer class; this does not imply a cast from Double or another integer width.",
}];

pub(super) const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "m.Data(indices) = typed_integer_rhs",
        inputs: &INTEGER_RHS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Exact integer bytes are encoded without floating-point conversion; incompatible class or shape fails before a mapped write.",
    }];
