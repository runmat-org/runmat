use crate::*;

const INTERP2_INTEGER_SAMPLE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X, Y, or V",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer grids and values must be exactly representable before interpolation.",
    }];
const INTERP2_INTEGER_QUERY_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Xq or Yq",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer query coordinates must be exactly representable before interpolation.",
    }];
const INTERP2_INTEGER_EXTRAPOLATION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "extrapolation value",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The scalar fill value must be exactly representable before interpolation.",
    }];
pub const INTERP2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = interp2(integer_X_Y_or_V, ...)",
        inputs: &INTERP2_INTEGER_SAMPLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer samples cross one checked binary64 boundary; MATLAB-compatible modes retain documented single/double grids and values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = interp2(X, Y, V, integer_Xq_or_Yq)",
        inputs: &INTERP2_INTEGER_QUERY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer queries are gated before provider access and converted only after exactness validation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = interp2(..., method, integer_extrapolation_value)",
        inputs: &INTERP2_INTEGER_EXTRAPOLATION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "RunMat accepts this broader scalar fill form only after compatibility and exactness checks that precede provider access or gather.",
    },
];
