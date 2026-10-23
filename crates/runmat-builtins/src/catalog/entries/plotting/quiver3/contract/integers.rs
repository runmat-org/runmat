use crate::*;

pub(super) const QUIVER3_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X/Y/Z",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Three-dimensional Quiver coordinate data accept numeric arrays. RunMat retains native class, shape, and exact values as graphics-object source properties.",
    },
    BuiltinIntegerInputCapability {
        name: "U/V/W",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Vector components remain authoritative in native storage until the explicit floating renderer boundary.",
    },
];

pub const QUIVER3_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "h = quiver3([integer_X,integer_Y,]integer_Z,integer_U,integer_V,integer_W,...)",
        inputs: &QUIVER3_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native storage is authoritative for XData/YData/ZData/UData/VData/WData and scene persistence. Rendering is an explicit floating geometry boundary; documented gpuArray input executes client-side and gathers through its exact owner.",
    }];
