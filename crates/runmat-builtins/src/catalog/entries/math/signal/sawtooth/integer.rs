use crate::*;
const SAWTOOTH_INTEGER_T_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "t",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer sample times are outside the documented double input domain and are admitted only when every value is exactly representable as binary64.",
    }];
const SAWTOOTH_INTEGER_T_XMAX_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "t",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer sample times cross the checked binary64 waveform boundary only in RunMat mode.",
    },
    BuiltinIntegerInputCapability {
        name: "xmax",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The typed peak-position control is checked exactly before conversion and must remain in the documented interval from zero through one.",
    },
];
pub const SAWTOOTH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = sawtooth(integer_t)",
        inputs: &SAWTOOTH_INTEGER_T_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The RunMat-only form returns double waveform samples and automatic residency may gather through the exact owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = sawtooth(integer_t, integer_xmax)",
        inputs: &SAWTOOTH_INTEGER_T_XMAX_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each independently typed argument is compatibility-gated and proved binary64-exact before evaluation or provider access.",
    },
];
