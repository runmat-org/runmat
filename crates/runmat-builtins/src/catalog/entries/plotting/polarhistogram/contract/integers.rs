use crate::*;

pub(super) const POLARHISTOGRAM_INTEGER_THETA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "theta or selected table data",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly documents all eight integer classes for direct theta data and any numeric class for the selected table variable.",
    }];

pub(super) const POLARHISTOGRAM_INTEGER_BINS_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nbins",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public positive-integer bin count is decoded structurally and bounded by the implementation's allocation limits.",
    }];

pub(super) const POLARHISTOGRAM_INTEGER_EDGE_COUNT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "edges, counts, or numeric bin properties",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public bin edges, manual counts, limits, widths, and numeric style properties accept numeric values and cross explicit histogram/rendering boundaries.",
    }];

pub(super) const POLARHISTOGRAM_INTEGER_SELECTOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "datavar variable index",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target permits numeric table-variable indices; these are structural one-based selectors rather than histogram observations.",
    }];

pub const POLARHISTOGRAM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = polarhistogram(integer_theta,___) or polarhistogram(table_with_integer_data,datavar,___)",
        inputs: &POLARHISTOGRAM_INTEGER_THETA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Authoritative integer samples gather exactly; angular binning and wedge rendering are explicit floating geometry/statistical boundaries.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = polarhistogram(theta,integer_nbins)",
        inputs: &POLARHISTOGRAM_INTEGER_BINS_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The count selects histogram structure and never becomes sample data.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = polarhistogram(theta,integer_edges) or polarhistogram(BinEdges=integer_edges,BinCounts=integer_counts,___)",
        inputs: &POLARHISTOGRAM_INTEGER_EDGE_COUNT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native values remain authoritative through parsing; histogram counts and rendered coordinates are explicit double outputs of this plotting sink.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = polarhistogram(tbl,integer_datavar,___)",
        inputs: &POLARHISTOGRAM_INTEGER_SELECTOR_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Table variable selection is a documented structural role; broader table-form support remains a general surface concern, not an implicit numeric conversion.",
    },
];
