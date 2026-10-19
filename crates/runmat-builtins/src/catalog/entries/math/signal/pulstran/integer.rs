use crate::*;
const PULSTRAN_INTEGER_T_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "T",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public page does not document typed-integer sample times; RunMat admits them only at an exact binary64 waveform boundary.",
    }];
const PULSTRAN_INTEGER_D_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "D",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "Typed delays and optional gains are independently gated before waveform evaluation.",
    }];
const PULSTRAN_INTEGER_P_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "P/FS/built-in pulse parameters",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed sampled prototypes and numeric controls cross checked floating interpolation or pulse-generation boundaries; callback extra arguments remain exact pass-through values.",
    }];
pub const PULSTRAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = pulstran(integer_T,D,FUN,...) or pulstran(integer_T,D,P,FS)",
        inputs: &PULSTRAN_INTEGER_T_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Sample times are checked before provider access and converted once for waveform evaluation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = pulstran(T,integer_D,FUN,...) or pulstran(T,integer_D,P,FS)",
        inputs: &PULSTRAN_INTEGER_D_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Delay and gain values are checked independently before summation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = pulstran(T,D,integer_P,integer_FS) or built-in pulse with integer parameters",
        inputs: &PULSTRAN_INTEGER_P_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each typed numeric role is admitted separately; function-handle callback payloads are not reclassified as pulstran computation controls.",
    },
];
