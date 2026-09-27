use crate::*;
const HISTC_DOCUMENTED_CLASSES: [BuiltinIntegerClass; 6] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
    BuiltinIntegerClass::Uint32,
];
const HISTC_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &HISTC_DOCUMENTED_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "MATLAB documents integer classes through 32 bits; int64 and uint64 are gated RunMat extensions.",
    },
    BuiltinIntegerInputCapability {
        name: "binranges",
        classes: &HISTC_DOCUMENTED_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "MATLAB documents integer classes through 32 bits for bin ranges.",
    },
];
const HISTC_WIDE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x_or_binranges",
        classes: &[
            BuiltinIntegerClass::Int64,
            BuiltinIntegerClass::Uint64,
        ],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "int64 and uint64 are not documented for legacy histc and require the wide-integer extension gate.",
    }];
pub const HISTC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[bincounts, ind] = histc(integer_x, integer_binranges, dim)",
        inputs: &HISTC_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented integer inputs are host-only and produce double counts and indices; unsupported classes and residency gate before conversion or gather.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[bincounts, ind] = histc(wide_integer_x_or_binranges, ...)",
        inputs: &HISTC_WIDE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "A gated RunMat extension; values enter the legacy floating histogram domain only after the compatibility gate.",
    },
];
