use crate::*;

pub(super) const SCATTER_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X/Y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Coordinate data accept all eight integer classes and retain native class, shape, and exact values as graphics-object source properties.",
    },
    BuiltinIntegerInputCapability {
        name: "S/C rendering data",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Marker areas and numeric color data accept all integer classes and cross explicitly into floating rendering metadata.",
    },
];

pub const SCATTER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "h = scatter(integer_X, integer_Y [, integer_S, integer_C], ...)",
        inputs: &SCATTER_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "XData/YData and figure persistence retain native storage. Marker sizing, colormap normalization, and rendering are explicit floating boundaries; integer gpuArray coordinates gather exactly because the direct WGPU scatter path is floating-only.",
    }];
