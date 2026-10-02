use crate::*;

const INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "numerator", classes: &crate::catalog::integer::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "The compatibility target documents single and double coefficients. RunMat mode admits exactly binary64-representable real integer coefficients." },
    BuiltinIntegerInputCapability { name: "denominator", classes: &crate::catalog::integer::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "The compatibility target documents single and double coefficients. RunMat mode admits exactly binary64-representable real integer coefficients." },
];
pub const DECONV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "[Q,R] = deconv(integer_numerator, integer_denominator)", inputs: &INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::Double,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "Each authoritative coefficient is checked for exact binary64 representation before division. Resident inputs gather through their owners and requested results return to the first resident owner.",
}];
