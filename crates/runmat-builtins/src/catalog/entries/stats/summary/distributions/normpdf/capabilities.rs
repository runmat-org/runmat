use crate::*;

pub const NORMPDF_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normpdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normpdf with typed-integer evaluation values is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormpdfIntegerXExtension"),
};
pub const NORMPDF_INTEGER_MU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normpdf-integer-mu",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normpdf with typed-integer mean parameters is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormpdfIntegerMuExtension"),
};
pub const NORMPDF_INTEGER_SIGMA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "normpdf-integer-sigma",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "normpdf with typed-integer standard-deviation parameters is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:NormpdfIntegerSigmaExtension"),
    };
pub const NORMPDF_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "normpdf-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "normpdf with typed-logical logical numeric inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:NormpdfLogicalInputExtension"),
    };
pub const NORMPDF_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    NORMPDF_INTEGER_X_EXTENSION,
    NORMPDF_INTEGER_MU_EXTENSION,
    NORMPDF_INTEGER_SIGMA_EXTENSION,
    NORMPDF_LOGICAL_INPUT_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability { name: "x or p", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Typed-integer evaluation values are gated independently and require exact binary64 representation." },
    BuiltinIntegerInputCapability { name: "mu", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Typed-integer mean parameters are gated independently and require exact binary64 representation." },
    BuiltinIntegerInputCapability { name: "sigma", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Typed-integer standard deviations are gated independently and require exact binary64 representation." },
];
pub const NORMPDF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "normpdf(integer_x_or_p, integer_mu, integer_sigma)", inputs: &INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Typed-integer inputs are RunMat-only, cross one checked binary64 distribution boundary, and return single only when another documented input is single; fallback restores through the exact owner when it can preserve the required class, otherwise automatic residency may remain host and explicit residency errors." }];
