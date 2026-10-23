use crate::*;

pub(super) const BAR_Y_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Y",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Y explicitly accepts every built-in integer class as scalar, vector, or matrix bar-height data.",
}];

pub(super) const BAR_X_Y_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "X explicitly accepts every built-in integer class as category-position data.",
    },
    BuiltinIntegerInputCapability {
        name: "Y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Y independently accepts every built-in integer class as scalar, vector, or matrix bar-height data.",
    },
];

pub(super) const BAR_WIDTH_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "width",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The positional width explicitly accepts every built-in integer class as a real numeric scalar.",
    }];

pub(super) const BAR_LINE_WIDTH_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "LineWidth",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes:
            "LineWidth explicitly accepts every built-in integer class as a real numeric scalar.",
    }];

pub const BAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "b = bar(integer_Y)",
        inputs: &BAR_Y_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Authoritative integer Y storage crosses one explicit client graphics boundary. Resident integer data gathers exactly before conversion because the floating shared-WGPU vertex path cannot reinterpret native integer buffers; the output is one or more opaque Bar handles.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "b = bar(integer_X, integer_Y)",
        inputs: &BAR_X_Y_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "X remains exact through category-count and uniqueness validation, including wide int64 and uint64 labels, before the deliberate client graphics conversion. Resident X and integer Y inputs gather through the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "b = bar(..., integer_width)",
        inputs: &BAR_WIDTH_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The positional width reads exactly one host numeric scalar and then crosses the explicit f32 graphics-property boundary; it does not produce integer data.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "b = bar(..., \"LineWidth\", integer_width)",
        inputs: &BAR_LINE_WIDTH_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "LineWidth reads exactly one host numeric scalar and then crosses the explicit f32 graphics-property boundary; it does not produce integer data.",
    },
];
