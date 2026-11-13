use crate::*;

const INTERP1_INTEGER_SAMPLE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X or Y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "All values must be exactly representable before the binary64 interpolation boundary.",
    }];
const INTERP1_INTEGER_QUERY_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Xq",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All coordinates must be exactly representable before the binary64 interpolation boundary.",
    }];
const INTERP1_INTEGER_EXTRAPOLATION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "extrapolation value",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The scalar fill value must be exactly representable before interpolation.",
    }];
pub const INTERP1_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = interp1(integer_X_or_Y, ..., Xq)",
        inputs: &INTERP1_INTEGER_SAMPLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat retains typed-integer samples as a checked extension; MATLAB-compatible modes retain the documented floating and temporal surface.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = interp1(X, Y, integer_Xq)",
        inputs: &INTERP1_INTEGER_QUERY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer queries are classified before provider dispatch or gather and then cross one checked binary64 boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = interp1(..., method, integer_extrapolation_value)",
        inputs: &INTERP1_INTEGER_EXTRAPOLATION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "RunMat accepts this broader scalar fill form only after compatibility and exactness checks that precede provider access or gather.",
    },
];
