use crate::*;

pub(super) const PLOT3_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X/Y/Z",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Three-dimensional coordinate data accept all eight integer classes and retain native class, shape, and exact values as graphics-object source properties.",
    }];

pub const PLOT3_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "h = plot3(integer_X, integer_Y, integer_Z, ...)",
        inputs: &PLOT3_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native numeric storage remains authoritative for XData/YData/ZData and figure persistence. Rendering is an explicit floating geometry boundary; integer gpuArray inputs gather exactly because the direct WGPU line path is floating-only.",
    }];
