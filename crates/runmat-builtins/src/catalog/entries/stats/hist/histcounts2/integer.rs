use crate::*;

const HISTCOUNTS2_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X_and_Y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented; X and Y may use different classes but must have identical sizes.",
    },
    BuiltinIntegerInputCapability {
        name: "bin_controls",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed NumBins and explicit integer edges follow their scalar/vector contracts; same-class wide explicit edges are classified exactly.",
    },
];
pub const HISTCOUNTS2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[N, Xedges, Yedges, binX, binY] = histcounts2(integer_X, integer_Y, ...)",
        inputs: &HISTCOUNTS2_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Counts, returned edges, and bin indices are double; source-domain exact comparisons are retained for same-class explicit integer edges before that output boundary.",
    }];
