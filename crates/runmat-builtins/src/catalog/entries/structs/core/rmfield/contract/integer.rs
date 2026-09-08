use crate::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};

const NESTED_VALUES: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "integer values nested in S",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Retained fixed-width integer values preserve their native class and payload.",
}];

pub const RMFIELD_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "S2 = rmfield(S_with_integer_fields, fields)",
        inputs: NESTED_VALUES,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "rmfield changes only structure metadata. Retained host and resident values move with the structure without numeric conversion or provider access.",
    }];
