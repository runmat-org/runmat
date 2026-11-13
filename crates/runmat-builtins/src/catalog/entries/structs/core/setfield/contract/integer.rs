use crate::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};

const VALUE: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "value",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Direct field assignment preserves native integer storage.",
}];
const SELECTOR: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "idx",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Numeric selector cells accept exact positive integer values.",
}];
pub const SETFIELD_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[
    BuiltinIntegerCapabilityDescriptor { form: "S = setfield(S, field, integer_value)", inputs: VALUE, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Replacing a field does not numerically convert its payload." },
    BuiltinIntegerCapabilityDescriptor { form: "S = setfield(S, {integer_idx}, field, ..., value)", inputs: SELECTOR, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Selectors are decoded to checked one-based positions; indexed assignment follows the target container's conversion rules." },
];
