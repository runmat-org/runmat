use crate::*;
const TRIPULS_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "T",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented sample-time domain is single or double. RunMat mode admits typed integers only when every value is exactly representable at the binary64 pulse boundary.",
    }];
const TRIPULS_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "W",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed width is a mode-gated scalar and must convert exactly before positive-range validation.",
    },
    BuiltinIntegerInputCapability {
        name: "S",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed skew is a mode-gated scalar and must convert exactly before validation in the closed interval from -1 through 1.",
    },
];
pub const TRIPULS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = tripuls(integer_T, ...)",
        inputs: &TRIPULS_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Compatibility and exactness checks occur before provider access; admitted integer samples produce a same-shaped double waveform.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = tripuls(T, integer_W, integer_S?)",
        inputs: &TRIPULS_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Width and skew cross checked scalar binary64 boundaries; output shape, class, and residency follow the supported sample-time path.",
    },
];
