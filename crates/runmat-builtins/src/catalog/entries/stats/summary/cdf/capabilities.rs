use crate::*;

pub const CDF_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cdf with typed-integer evaluation points is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CdfIntegerXExtension"),
};

pub const CDF_INTEGER_PARAMETER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cdf-integer-parameters",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cdf with typed-integer distribution parameters is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CdfIntegerParametersExtension"),
    };

pub const CDF_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cdf-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cdf with logical numeric inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CdfLogicalInputExtension"),
};

pub const CDF_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    CDF_INTEGER_X_EXTENSION,
    CDF_INTEGER_PARAMETER_EXTENSION,
    CDF_LOGICAL_INPUT_EXTENSION,
];

const CDF_INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer evaluation points are gated before provider access and must be exactly representable at the floating CDF boundary.",
    }];

const CDF_INTEGER_PARAMETER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A...D",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer named-distribution parameters are independently gated and must be exactly representable at the floating CDF boundary.",
    }];

pub const CDF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "p = cdf(pd, x) or cdf(name, x, A...) with integer x",
        inputs: &CDF_INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only typed-integer x values produce double probabilities unless a named-form single parameter makes the result single; resident fallback restores output to the first owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "p = cdf(name, x, A...) with integer distribution parameters",
        inputs: &CDF_INTEGER_PARAMETER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only typed-integer parameters preserve authoritative storage through admission and scalar expansion, then cross an exact binary64 boundary for evaluation.",
    },
];
