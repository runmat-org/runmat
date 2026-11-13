use crate::*;

pub(super) const STEM_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X and Y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly lists all eight integer classes for both coordinate roles. Native class, shape, and exact values remain authoritative as graphics-object source data.",
    }];

pub const STEM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "h = stem(integer_Y) or stem(integer_X, integer_Y, ...)",
        inputs: &STEM_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "XData and YData retain native storage while rendering deliberately materializes floating geometry. Floating gpuArray inputs can use the direct WGPU sink; integer resident inputs gather exactly through their owner.",
    }];
