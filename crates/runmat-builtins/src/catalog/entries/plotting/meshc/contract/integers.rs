use crate::*;

pub(super) const INTEGER_X: [BuiltinIntegerInputCapability; 1] = [meshc_integer_input("X")];

pub(super) const INTEGER_Y: [BuiltinIntegerInputCapability; 1] = [meshc_integer_input("Y")];

pub(super) const INTEGER_Z: [BuiltinIntegerInputCapability; 1] = [meshc_integer_input("Z")];

pub(super) const fn meshc_integer_input(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "The compatibility target explicitly lists every built-in integer class for this surface coordinate role.",
    }
}

pub(super) const fn meshc_integer_capability(
    form: &'static str,
    inputs: &'static [BuiltinIntegerInputCapability],
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native class, exact values, and shape remain authoritative on the primary Surface object and in RunMat scene persistence; surface rendering, contour generation, and client-side GPU handling are explicit floating boundaries. The documented independent C color-array form is a general unimplemented graphics gap.",
    }
}

pub const MESHC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    meshc_integer_capability("s = meshc(integer_X, Y, Z, ...)", &INTEGER_X),
    meshc_integer_capability("s = meshc(X, integer_Y, Z, ...)", &INTEGER_Y),
    meshc_integer_capability("s = meshc(X, Y, integer_Z, ...)", &INTEGER_Z),
];
