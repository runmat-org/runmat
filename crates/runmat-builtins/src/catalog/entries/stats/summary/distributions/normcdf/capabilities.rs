use crate::*;

pub const NORMCDF_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normcdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normcdf with typed-integer evaluation values is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormcdfIntegerXExtension"),
};
pub const NORMCDF_INTEGER_MU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normcdf-integer-mu",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normcdf with typed-integer mean parameters is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormcdfIntegerMuExtension"),
};
pub const NORMCDF_INTEGER_SIGMA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "normcdf-integer-sigma",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "normcdf with typed-integer standard-deviation parameters is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:NormcdfIntegerSigmaExtension"),
    };
pub const NORMCDF_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "normcdf-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "normcdf with typed-logical logical numeric inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:NormcdfLogicalInputExtension"),
    };
pub const NORMCDF_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    NORMCDF_INTEGER_X_EXTENSION,
    NORMCDF_INTEGER_MU_EXTENSION,
    NORMCDF_INTEGER_SIGMA_EXTENSION,
    NORMCDF_LOGICAL_INPUT_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability { name: "x or p", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Typed-integer evaluation values are gated independently and require exact binary64 representation." },
    BuiltinIntegerInputCapability { name: "mu", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Typed-integer mean parameters are gated independently and require exact binary64 representation." },
    BuiltinIntegerInputCapability { name: "sigma", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Typed-integer standard deviations are gated independently and require exact binary64 representation." },
];
pub const NORMCDF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "normcdf(integer_x_or_p, integer_mu, integer_sigma)", inputs: &INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Typed-integer inputs are RunMat-only, cross one checked binary64 distribution boundary, and return single only when another documented input is single; fallback restores through the exact owner when it can preserve the required class, otherwise automatic residency may remain host and explicit residency errors." }];
