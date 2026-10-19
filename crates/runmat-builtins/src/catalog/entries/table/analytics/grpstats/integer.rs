use crate::*;

const GRPSTATS_INTEGER_GROUP_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "group",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Statistics grouping variables explicitly admit signed and unsigned integer classes and are compared exactly.",
    }];
const GRPSTATS_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented matrix X datatype list excludes signed and unsigned integer data; RunMat admission is independently mode-gated.",
    }];
const GRPSTATS_INTEGER_TABLE_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer table data variables",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Table variables may be numeric, so selected native integer variables use the documented table form.",
    }];
pub const GRPSTATS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "stats = grpstats(X,integer_group,___)",
        inputs: &GRPSTATS_INTEGER_GROUP_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer grouping keys remain exact, including above flintmax; summary output is governed by X and whichstats.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "stats = grpstats(integer_X,group,___)",
        inputs: &GRPSTATS_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer matrix data crosses a checked binary64 boundary after its compatibility gate; values that cannot be represented exactly reject rather than silently rounding.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "tblstats = grpstats(tbl,groupvars,___) with integer DataVars",
        inputs: &GRPSTATS_INTEGER_TABLE_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Selected integer table variables are documented numeric data; extrema retain exact integer values while floating statistics enter their explicit computation domain.",
    },
];
