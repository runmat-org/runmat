use crate::*;

pub(super) const HIST_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "MATLAB documents single, double, logical, and categorical sample data for legacy hist; integer samples are compatibility-gated.",
    }];

pub(super) const HIST_INTEGER_BIN_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nbins",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All integer classes are documented for scalar nbins.",
    }];

pub(super) const HIST_INTEGER_CENTERS_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "xbins",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "MATLAB documents single or double numeric center vectors; typed integer centers are compatibility-gated.",
    }];

pub const HIST_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[counts, centers] = hist(integer_X, ...)",
        inputs: &HIST_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "A gated RunMat extension; integer-center forms compare original typed observations against exact half-integer midpoints before Double outputs. Other bin forms retain their floating computation domain; all legacy hist outputs are double counts and centers.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[counts, centers] = hist(X, integer_nbins)",
        inputs: &HIST_INTEGER_BIN_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Scalar integer nbins is documented and parsed exactly before conversion to a platform bin count.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[counts, centers] = hist(X, integer_xbins)",
        inputs: &HIST_INTEGER_CENTERS_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer center vectors are a gated RunMat extension. Order, spacing, and half-integer midpoint geometry remain exact; original integer observations are classified before conversion. Weights, normalization, chart widths, and declared Double center outputs use floating arithmetic only after the exact boundary decisions.",
    },
];
