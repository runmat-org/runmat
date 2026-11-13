use crate::*;

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Host data accepts all eight integer classes; interactive GPU data accepts integer classes through 32 bits and rejects int64/uint64 before gather.",
    },
    BuiltinIntegerInputCapability {
        name: "bins_and_numeric_controls",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed integer bin counts, explicit edges, widths, and limits are parsed according to each control's scalar/vector contract.",
    },
];

pub const HISTCOUNTS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[N, edges] = histcounts(X, bins/name-value controls)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer observations enter the documented double histogram domain and counts/edges are double; integer NumBins parsing remains exact and 64-bit integer gpuArray data is unsupported.",
    }];
