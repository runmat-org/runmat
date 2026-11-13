use crate::*;

pub(super) const HISTOGRAM2_INTEGER_DATA: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X and Y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Paired histogram observations accept every integer class and are retained as the chart Data matrix using the language concatenation dominance rule.",
    }];

pub(super) const HISTOGRAM2_INTEGER_EDGES: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "XBinEdges and YBinEdges",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes:
            "Explicit integer edge vectors are classified exactly before renderer materialization.",
    }];

pub const HISTOGRAM2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = histogram2(integer_X, integer_Y, ...)",
        inputs: &HISTOGRAM2_INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Authoritative typed observations are retained on the chart while counts and rendering use explicit double boundaries.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = histogram2(X, Y, integer_XBinEdges, integer_YBinEdges)",
        inputs: &HISTOGRAM2_INTEGER_EDGES,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Wide same-class integer edges remain exact during bin membership classification.",
    },
];
