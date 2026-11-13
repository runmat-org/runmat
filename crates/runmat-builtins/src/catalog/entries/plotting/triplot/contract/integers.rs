use crate::*;

pub(super) const TRIPLOT_INTEGER_CONNECTIVITY_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "TRI or ConnectivityList",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target describes a connectivity matrix without enumerating native integer storage. RunMat mode decodes every vertex ID exactly as a positive one-based structural index.",
    }];

pub(super) const TRIPLOT_INTEGER_COORDINATE_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed X coordinates are mode-gated and must be exactly representable at the binary64 plotting boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "Y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed Y coordinates are mode-gated and must be exactly representable at the binary64 plotting boundary.",
    },
];

pub const TRIPLOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = triplot(integer_TRI, X, Y, ...)",
        inputs: &TRIPLOT_INTEGER_CONNECTIVITY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Connectivity remains exact through bounds validation and only selects coordinate elements; the returned graphics handle is double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = triplot(TRI, integer_X, integer_Y, ...)",
        inputs: &TRIPLOT_INTEGER_COORDINATE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Coordinates cross one checked binary64 rendering boundary after compatibility admission; triplot is a host plotting sink.",
    },
];
