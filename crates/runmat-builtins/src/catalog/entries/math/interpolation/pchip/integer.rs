use crate::*;

const PCHIP_INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double sample locations. RunMat admits typed integers only when each value is exactly representable at the binary64 interpolation boundary.",
    }];
const PCHIP_INTEGER_Y_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double sample values. RunMat admits typed integers only through the explicit floating interpolation boundary.",
    }];
const PCHIP_INTEGER_XQ_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Xq",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double query points. RunMat admits typed integers only when exact binary64 conversion is possible.",
    }];
pub const PCHIP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "pp = pchip(integer_X, Y) or Vq = pchip(integer_X, Y, Xq)",
        inputs: &PCHIP_INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed sample locations are independently gated before gather and cross once into the floating PCHIP coefficient domain.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "pp = pchip(X, integer_Y) or Vq = pchip(X, integer_Y, Xq)",
        inputs: &PCHIP_INTEGER_Y_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed sample values are independently gated before gather and produce floating coefficients or interpolated values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = pchip(X, Y, integer_Xq)",
        inputs: &PCHIP_INTEGER_XQ_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed query points are independently gated and converted only after exactness is proved.",
    },
];
