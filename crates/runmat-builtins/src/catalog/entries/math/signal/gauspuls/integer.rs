use crate::*;

const GAUSPULS_INTEGER_TIME_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "T",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are read from authoritative storage and admitted only when exactly representable as binary64 sample times.",
    }];

const GAUSPULS_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "FC, BW, BWR, or TPE",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are scalar-only and must be exactly representable at the binary64 signal-computation boundary.",
    }];

pub const GAUSPULS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = gauspuls(integer_T, FC, BW, BWR)",
        inputs: &GAUSPULS_INTEGER_TIME_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat-only integer sample times cross one exact binary64 boundary and produce host double output.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y or TC = gauspuls(..., integer_control, ...)",
        inputs: &GAUSPULS_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "RunMat-only scalar controls cross one exact binary64 boundary; output follows the documented floating form.",
    },
];
