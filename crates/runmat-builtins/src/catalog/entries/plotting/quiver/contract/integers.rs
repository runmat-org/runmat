use crate::*;

pub(super) const QUIVER_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X/Y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Quiver coordinate data accept numeric arrays, and the public Quiver property contract includes integer numeric types. RunMat retains native class, shape, and exact values as graphics-object source properties.",
    },
    BuiltinIntegerInputCapability {
        name: "U/V",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Vector components follow the same numeric graphics-data contract and remain authoritative until the explicit floating renderer boundary.",
    },
];

pub const QUIVER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "h = quiver([integer_X,integer_Y,]integer_U,integer_V,...)",
        inputs: &QUIVER_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native storage is authoritative for XData/YData/UData/VData and scene persistence. Rendering is an explicit floating geometry boundary; documented gpuArray input executes client-side and integer buffers gather exactly when the direct floating WGPU path is inapplicable.",
    }];
