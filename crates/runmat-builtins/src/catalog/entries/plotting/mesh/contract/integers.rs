use crate::*;

pub(super) const INTEGER_X: [BuiltinIntegerInputCapability; 1] = [mesh_integer_input("X")];

pub(super) const INTEGER_Y: [BuiltinIntegerInputCapability; 1] = [mesh_integer_input("Y")];

pub(super) const INTEGER_Z: [BuiltinIntegerInputCapability; 1] = [mesh_integer_input("Z")];

pub(super) const fn mesh_integer_input(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "The compatibility target explicitly lists every built-in integer class for this surface coordinate role.",
    }
}

pub(super) const fn mesh_integer_capability(
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
        notes: "Native class, exact values, and shape remain authoritative on XData/YData/ZData and in RunMat scene persistence; rendering and client-side GPU handling are explicit floating boundaries. The documented independent C color-array form is a general unimplemented graphics gap, not an integer coercion rule.",
    }
}

pub const MESH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    mesh_integer_capability("s = mesh(integer_X, Y, Z, ...)", &INTEGER_X),
    mesh_integer_capability("s = mesh(X, integer_Y, Z, ...)", &INTEGER_Y),
    mesh_integer_capability("s = mesh(X, Y, integer_Z, ...)", &INTEGER_Z),
];
