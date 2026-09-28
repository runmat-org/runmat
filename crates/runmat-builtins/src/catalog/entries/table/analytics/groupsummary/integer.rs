use crate::*;

const GROUPSUMMARY_INTEGER_GROUP_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer grouping variables",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer table grouping variables are compared exactly and preserve class in the output table.",
    }];
const GROUPSUMMARY_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer data variables",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer table data variables are numeric and are valid data-variable selections; exact-class extrema and explicit floating statistics preserve the documented method contract.",
    }];
const GROUPSUMMARY_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "native-class integer bin count or boolean name-value control",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Native integer scalars in these structurally ambiguous control positions are accepted only in RunMat extension mode.",
    }];
pub const GROUPSUMMARY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "groupsummary(T_or_A,integer_groupvars,method,___)",
        inputs: &GROUPSUMMARY_INTEGER_GROUP_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Table and array grouping is exact. Table grouping columns and array BG preserve class, while GroupCount/BC and summaries are function-specific.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "groupsummary(integer_A,groupvars,method) or table integer data variables",
        inputs: &GROUPSUMMARY_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented integer array and table data uses exact native-class min and max plus explicit floating computation for mean, sum, median, and counts. One-group numeric-edge and scalar-count forms compare integers exactly; multi-group bin specifications, time bins, additional named methods, and function handles currently reject explicitly.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "groupsummary(T,groupvars,integer_numbins,method,___) or integer boolean controls",
        inputs: &GROUPSUMMARY_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "This is an explicitly gated input-form extension; equivalent scalar-double and logical controls remain available in strict compatibility mode.",
    },
];
