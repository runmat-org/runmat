use crate::*;

pub(super) const SURFC_INTEGER_X: [BuiltinIntegerInputCapability; 1] = [surfc_integer_input("X")];

pub(super) const SURFC_INTEGER_Y: [BuiltinIntegerInputCapability; 1] = [surfc_integer_input("Y")];

pub(super) const SURFC_INTEGER_Z: [BuiltinIntegerInputCapability; 1] = [surfc_integer_input("Z")];

pub(super) const fn surfc_integer_input(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "The compatibility target explicitly lists every native integer class for this surface coordinate role.",
    }
}

pub(super) const fn surfc_integer_capability(
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
        notes: "Native class, exact values, and shape remain authoritative in scene XData/YData/ZData; surface rendering and projected-contour construction are explicit floating boundaries.",
    }
}

pub const SURFC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    surfc_integer_capability("h = surfc(integer_X, Y, Z, ...)", &SURFC_INTEGER_X),
    surfc_integer_capability("h = surfc(X, integer_Y, Z, ...)", &SURFC_INTEGER_Y),
    surfc_integer_capability("h = surfc(X, Y, integer_Z, ...)", &SURFC_INTEGER_Z),
];
