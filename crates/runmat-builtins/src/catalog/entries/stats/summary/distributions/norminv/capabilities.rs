use crate::*;

pub const NORMINV_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "norminv-integer-p",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "norminv with typed-integer evaluation values is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NorminvIntegerPExtension"),
};
pub const NORMINV_INTEGER_MU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "norminv-integer-mu",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "norminv with typed-integer mean parameters is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NorminvIntegerMuExtension"),
};
pub const NORMINV_INTEGER_SIGMA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "norminv-integer-sigma",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "norminv with typed-integer standard-deviation parameters is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:NorminvIntegerSigmaExtension"),
    };
pub const NORMINV_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "norminv-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "norminv with typed-logical logical numeric inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:NorminvLogicalInputExtension"),
    };
pub const NORMINV_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    NORMINV_INTEGER_X_EXTENSION,
    NORMINV_INTEGER_MU_EXTENSION,
    NORMINV_INTEGER_SIGMA_EXTENSION,
    NORMINV_LOGICAL_INPUT_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability { name: "x or p", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Typed-integer evaluation values are gated independently and require exact binary64 representation." },
    BuiltinIntegerInputCapability { name: "mu", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Typed-integer mean parameters are gated independently and require exact binary64 representation." },
    BuiltinIntegerInputCapability { name: "sigma", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Typed-integer standard deviations are gated independently and require exact binary64 representation." },
];
pub const NORMINV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "norminv(integer_x_or_p, integer_mu, integer_sigma)", inputs: &INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Typed-integer inputs are RunMat-only, cross one checked binary64 distribution boundary, and return single only when another documented input is single; fallback restores through the exact owner when it can preserve the required class, otherwise automatic residency may remain host and explicit residency errors." }];
