use crate::*;

const FILTER_INTEGER_NUMERIC_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    BuiltinIntegerInputCapability {
        name: "b",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight real integer classes are documented for numerator coefficients and enter the floating filtering domain.",
    },
    BuiltinIntegerInputCapability {
        name: "a",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight real integer classes are documented for denominator coefficients and enter the floating filtering domain.",
    },
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer signals are documented and produce double output.",
    },
    BuiltinIntegerInputCapability {
        name: "zi",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer initial conditions are documented and produce floating final conditions.",
    },
];

const FILTER_INTEGER_DIM_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The filtering dimension is an exact positive integer scalar.",
    }];

pub const FILTER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[y, zf] = filter(integer_b, integer_a, integer_x, integer_zi, dim)",
        inputs: &FILTER_INTEGER_NUMERIC_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Authoritative integer storage is read before conversion to the documented floating filtering domain; resident integer inputs gather rather than entering floating provider kernels.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[y, zf] = filter(b, a, x, zi, integer_dim)",
        inputs: &FILTER_INTEGER_DIM_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed integer dimensions are parsed exactly before host indexing or provider dispatch.",
    },
];
