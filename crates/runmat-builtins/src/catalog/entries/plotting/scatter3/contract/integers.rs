use crate::*;

pub(super) const SCATTER3_INTEGER_COORDINATE_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents all eight integer coordinate classes; native class, shape, and exact values remain authoritative as XData.",
    },
    BuiltinIntegerInputCapability {
        name: "Y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Native integer YData remains authoritative independently of renderer geometry.",
    },
    BuiltinIntegerInputCapability {
        name: "Z",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Native integer ZData remains authoritative independently of renderer geometry.",
    },
];

pub(super) const SCATTER3_INTEGER_STYLE_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "S",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly documents all eight integer classes for marker areas; validated values enter the floating renderer-size boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "C",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Numeric colormap indices or RGB components are renderer controls; integer identity is used before the documented color-mapping boundary.",
    },
];

pub const SCATTER3_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = scatter3(integer_X, integer_Y, integer_Z, ...)",
        inputs: &SCATTER3_INTEGER_COORDINATE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Graphics source properties retain authoritative native storage while a derived f32 vertex cache is used only for rendering. Documented gpuArray input executes client-side; unsupported integer buffers gather exactly through their owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = scatter3(X,Y,Z,integer_S [, integer_C])",
        inputs: &SCATTER3_INTEGER_STYLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Marker areas and colors are validated in their native numeric class, then deliberately materialized into finite renderer metadata; the returned graphics handle is double.",
    },
];
