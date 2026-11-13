use crate::*;

pub(super) const INTEGER_X: [BuiltinIntegerInputCapability; 1] = coordinate_input("X");

pub(super) const INTEGER_Y: [BuiltinIntegerInputCapability; 1] = coordinate_input("Y");

pub(super) const INTEGER_Z: [BuiltinIntegerInputCapability; 1] = coordinate_input("Z");

pub(super) const INTEGER_C: [BuiltinIntegerInputCapability; 1] = coordinate_input("C");

const fn coordinate_input(name: &'static str) -> [BuiltinIntegerInputCapability; 1] {
    [BuiltinIntegerInputCapability {
        name,
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "MATLAB documents every built-in integer class for this patch data role; authoritative class and shape are retained before explicit renderer conversion.",
    }]
}

pub(super) const INTEGER_AX: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "ax",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "MATLAB axes are graphics objects; typed-integer numeric aliases are a gated RunMat representation extension.",
}];

pub const FILL3_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    fill3_capability("p = fill3(integer_X, Y, Z, C, ...)", &INTEGER_X),
    fill3_capability("p = fill3(X, integer_Y, Z, C, ...)", &INTEGER_Y),
    fill3_capability("p = fill3(X, Y, integer_Z, C, ...)", &INTEGER_Z),
    fill3_capability("p = fill3(X, Y, Z, integer_C, ...)", &INTEGER_C),
    BuiltinIntegerCapabilityDescriptor {
        form: "p = fill3(integer_ax, X, Y, Z, C, ...)",
        inputs: &INTEGER_AX,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility extension gate runs before axes selection or input gathering.",
    },
];

pub(super) const fn fill3_capability(
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
        notes: "Integer source data remains authoritative on the Patch object; client rendering crosses an explicit floating-point boundary.",
    }
}
