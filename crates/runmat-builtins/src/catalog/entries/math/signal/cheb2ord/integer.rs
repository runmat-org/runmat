use crate::*;

const INTEGER_FREQUENCY_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Wp",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer passband edges are outside the documented single/double domain and enter a checked double computation boundary only in RunMat mode.",
    },
    BuiltinIntegerInputCapability {
        name: "Ws",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer stopband edges are outside the documented single/double domain and enter a checked double computation boundary only in RunMat mode.",
    },
];
const INTEGER_ATTENUATION_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Rp",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer ripple is a RunMat-only control and must be exactly representable at the double computation boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "Rs",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer attenuation is a RunMat-only control and must be exactly representable at the double computation boundary.",
    },
];

pub const CHEB2ORD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[n, Wn] = cheb2ord(integer_Wp, integer_Ws, Rp, Rs, ...)",
        inputs: &INTEGER_FREQUENCY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer frequency edges are validated from authoritative storage before deliberate double filter-order computation; interactive resident input is independently gated.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[n, Wn] = cheb2ord(Wp, Ws, integer_Rp, integer_Rs, ...)",
        inputs: &INTEGER_ATTENUATION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer ripple and attenuation controls are range-checked exactly before deliberate double filter-order computation.",
    },
];
