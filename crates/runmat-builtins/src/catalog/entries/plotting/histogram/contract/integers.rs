use crate::*;

pub(super) const HISTOGRAM_INTEGER_DATA: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Data",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Histogram data accepts every built-in integer class and remains authoritative on the chart object.",
    }];

pub(super) const HISTOGRAM_INTEGER_EDGES: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "BinEdges",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Explicit integer edges are ordered exactly before the floating rendering boundary.",
    }];

pub const HISTOGRAM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = histogram(integer_Data, ...)",
        inputs: &HISTOGRAM_INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact typed Data is retained while raw double counts and renderer coordinates remain explicit statistical and client boundaries.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = histogram(X, integer_BinEdges)",
        inputs: &HISTOGRAM_INTEGER_EDGES,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Bin membership compares against authoritative explicit edges before chart geometry materializes as double.",
    },
];
