use crate::*;

const BUTTORD_INTEGER_FREQUENCY_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Wp",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented passband-edge datatypes are single and double; RunMat mode additionally accepts scalar or two-element typed-integer storage.",
    },
    BuiltinIntegerInputCapability {
        name: "Ws",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented stopband-edge datatypes are single and double; RunMat mode additionally accepts scalar or two-element typed-integer storage.",
    },
];
const BUTTORD_INTEGER_ATTENUATION_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Rp",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented passband-ripple datatypes are single and double; RunMat mode additionally accepts every scalar integer class.",
    },
    BuiltinIntegerInputCapability {
        name: "Rs",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented stopband-attenuation datatypes are single and double; RunMat mode additionally accepts every scalar integer class.",
    },
];

pub const BUTTORD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[n,Wn] = buttord(integer_Wp,integer_Ws,Rp,Rs,...)",
        inputs: &BUTTORD_INTEGER_FREQUENCY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer frequency edges cross the explicit binary64 Butterworth analysis boundary after compatibility and residency checks. Both n and Wn remain host double values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[n,Wn] = buttord(Wp,Ws,integer_Rp,integer_Rs,...)",
        inputs: &BUTTORD_INTEGER_ATTENUATION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer ripple and attenuation controls cross the explicit binary64 decibel-analysis boundary after compatibility and residency checks. Both n and Wn remain host double values.",
    },
];
