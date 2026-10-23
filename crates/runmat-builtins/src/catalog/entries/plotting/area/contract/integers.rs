use crate::*;

pub(super) const AREA_Y_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Y independently accepts every built-in integer class as vector or matrix data.",
    }];

pub(super) const AREA_X_Y_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes:
            "X independently accepts every built-in integer class as vector or matrix coordinates.",
    },
    BuiltinIntegerInputCapability {
        name: "Y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Y independently accepts every built-in integer class as vector or matrix data.",
    },
];

pub(super) const AREA_BASE_VALUE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "basevalue",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The positional or BaseValue name-value control accepts a real numeric scalar.",
    }];

pub(super) const AREA_LINE_WIDTH_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "LineWidth",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "LineWidth explicitly accepts every built-in integer class as a scalar point width.",
    }];

pub const AREA_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "a = area(integer_Y)",
        inputs: &AREA_Y_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "An implicit double X axis is generated from one through the vector length or matrix row count. Authoritative integer Y storage crosses one explicit client graphics boundary; resident integer data gathers because area accepts GPU arrays but executes on the client.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "a = area(integer_X, integer_Y)",
        inputs: &AREA_X_Y_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "X and Y retain authoritative typed storage through shape validation and exact integer X ordering before the deliberate client graphics conversion. The output is one or more opaque Area graphics handles rather than integer data.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "a = area(..., integer_basevalue)",
        inputs: &AREA_BASE_VALUE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The positional and BaseValue name-value forms read exactly one real numeric scalar and then cross the explicit f64 graphics-property boundary; a resident scalar gathers on the client.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "a = area(..., \"LineWidth\", integer_width)",
        inputs: &AREA_LINE_WIDTH_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "LineWidth is validated and converted once by the shared host graphics-style parser; it does not create an integer output. Applying and exposing the parsed outline width is a general Area graphics-property gap, not an integer-storage distinction.",
    },
];
