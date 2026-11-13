use crate::*;

const DIRECT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "A", classes: &crate::catalog::integer::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "All eight real and complex integer classes cross the floating convolution boundary before arithmetic." },
    BuiltinIntegerInputCapability { name: "B", classes: &crate::catalog::integer::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Mixed integer classes convert independently to the selected floating output domain." },
];
const SEPARABLE_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "u",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The column kernel accepts all eight real or complex integer classes.",
    },
    BuiltinIntegerInputCapability {
        name: "v",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The row kernel accepts all eight real or complex integer classes.",
    },
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The signal matrix accepts all eight real or complex integer classes.",
    },
];
pub const CONV2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "C = conv2(integer_A, integer_B, shape?)", inputs: &DIRECT_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Single input selects single result; otherwise the result is double. Integer inputs never produce integer output; resident integer inputs gather exactly before floating convolution." },
    BuiltinIntegerCapabilityDescriptor { form: "C = conv2(integer_u, integer_v, integer_A, shape?)", inputs: &SEPARABLE_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The separable form applies the same single-dominant floating output rule across all three numeric inputs and supports full/same/valid shapes." },
];
