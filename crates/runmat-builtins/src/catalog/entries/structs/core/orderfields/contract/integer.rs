use crate::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};

const NESTED_VALUES: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "integer values nested in S1",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "Field values retain their native class and payload while top-level metadata is reordered.",
}];
const PERMUTATION: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "P",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed integer positions are range-checked without conversion through binary64.",
}];

pub const ORDERFIELDS_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[
    BuiltinIntegerCapabilityDescriptor {
        form: "S = orderfields(S1_with_integer_fields, ...)", inputs: NESTED_VALUES,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Only the ordered field map changes; resident handles and host payloads move without provider access.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[S, Pout] = orderfields(S1, integer_P)", inputs: PERMUTATION,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Pout is a host double column vector; reordered field payloads retain their classes.",
    },
];
