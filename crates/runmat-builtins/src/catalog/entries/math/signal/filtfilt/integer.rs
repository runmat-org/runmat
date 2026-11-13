use crate::*;
const FILTFILT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "b",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB documents single and double coefficients; RunMat mode additionally accepts real integer coefficients.",
    },
    BuiltinIntegerInputCapability {
        name: "a",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB documents single and double coefficients; RunMat mode additionally accepts real integer coefficients.",
    },
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB documents single and double signals; RunMat mode additionally accepts real integer vectors.",
    },
];

pub const FILTFILT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "y = filtfilt(integer_b, integer_a, integer_x)",
        inputs: &FILTFILT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer inputs are independently mode-gated before gathering or provider access and produce double output.",
    }];
