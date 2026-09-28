use crate::*;

const GROUP_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "integer grouping or DataVariable table columns",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Grouping keys are compared exactly. Aggregation of a numeric data column follows the selected method's result domain.",
}];
const SELECTORS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "rowvars, colvars, or datavar variable indices",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Implemented positional selectors use exact one-based integer indices.",
}];

pub const PIVOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "P = pivot(table_with_integer_group_or_data_variables, rowvars, colvars, datavar, ___)",
        inputs: &GROUP_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer row keys retain exact storage; the selected aggregation currently constructs double summary columns.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "P = pivot(T, integer_rowvars, integer_colvars, integer_datavar, ___)",
        inputs: &SELECTORS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Implemented positional selector parsing reads integer storage directly. Newer named Rows/Columns/DataVariable syntax is not implemented.",
    },
];
