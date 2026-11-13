use crate::*;

pub(super) const fn input_capability(name: &'static str) -> [BuiltinIntegerInputCapability; 1] {
    [BuiltinIntegerInputCapability { name, classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "MATLAB documents every built-in integer class; exact class and shape remain authoritative on the Patch object before renderer conversion." }]
}

pub(super) const INTEGER_X: [BuiltinIntegerInputCapability; 1] = input_capability("X");

pub(super) const INTEGER_Y: [BuiltinIntegerInputCapability; 1] = input_capability("Y");

pub(super) const INTEGER_C: [BuiltinIntegerInputCapability; 1] = input_capability("C");

pub(super) const INTEGER_AX: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "ax",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "MATLAB axes are graphics objects; typed-integer numeric aliases are a gated RunMat representation extension.",
}];

pub(super) const fn capability(
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
        notes: "Patch data remains authoritative; client rendering is the explicit floating-point boundary.",
    }
}

pub const FILL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    capability("p = fill(integer_X, Y, C, ...)", &INTEGER_X),
    capability("p = fill(X, integer_Y, C, ...)", &INTEGER_Y),
    capability("p = fill(X, Y, integer_C, ...)", &INTEGER_C),
    BuiltinIntegerCapabilityDescriptor {
        form: "p = fill(integer_ax, X, Y, C, ...)",
        inputs: &INTEGER_AX,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility extension gate runs before axes selection or input gathering.",
    },
];
