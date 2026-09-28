use crate::*;

const STORED_VALUES: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "selected integer value or table variable",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Selection preserves the exact stored integer class and payload; no floating-point key conversion is used.",
}];
const ASSIGNED_VALUES: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "rhs",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer assignment uses the table variable or dictionary entry's storage contract and preserves exact wide values when admissible.",
}];

pub const TABLE_SUBSREF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "out = table.subsref(T_with_integer_variables, S)",
    inputs: &STORED_VALUES,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "Dot and brace selection retain the selected variable's integer class; parentheses selection returns a table containing typed variables.",
}];
pub const TABLE_SUBSASGN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "T = table.subsasgn(T, S, integer_rhs)",
    inputs: &ASSIGNED_VALUES,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "The returned table retains typed variables. Assignment checks dimensions and applies the target variable's conversion behavior.",
}];
pub const DICTIONARY_SUBSREF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "out = dictionary.subsref(d_with_integer_entries, S)",
    inputs: &STORED_VALUES,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "Integer key comparison and stored integer values retain their native classes, including wide int64 and uint64 keys.",
}];
pub const DICTIONARY_SUBSASGN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "d = dictionary.subsasgn(d, S, integer_rhs)",
    inputs: &ASSIGNED_VALUES,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "Assignments compare typed keys exactly and normalize new values to the existing dictionary value class.",
}];
