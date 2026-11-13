use crate::*;

const GRIDDED_INTEGER_GRID_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x, X_i, or gridVecs",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes must be exactly representable at the binary64 interpolation-coordinate boundary.",
    }];

const GRIDDED_INTEGER_VALUES_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "v, V, or Values",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes must be exactly representable before the object stores ordinary double Values.",
    }];

const GRIDDED_INTEGER_QUERY_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Xq, Xq_i, or query grid vectors",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All query syntaxes admit the same eight integer classes only when every coordinate is exactly representable as binary64.",
    }];

pub const GRIDDED_INTERPOLANT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "F = griddedInterpolant(integer_grid, V, ...)",
        inputs: &GRIDDED_INTEGER_GRID_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer sample grids cross one checked binary64 boundary; the constructor returns an interpolant object whose GridVectors are floating arrays.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "F = griddedInterpolant(grid, integer_V, ...)",
        inputs: &GRIDDED_INTEGER_VALUES_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer sample values cross one checked binary64 boundary, are stored as double Values, and produce double evaluations.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = F(integer_query)",
        inputs: &GRIDDED_INTEGER_QUERY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer query coordinates cross one checked binary64 boundary; output precision continues to follow the interpolant Values class.",
    },
];
