use crate::*;

pub const TINV_INTEGER_P_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tinv-integer-probability",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tinv with typed-integer probabilities is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TinvIntegerProbabilityExtension"),
};
pub const TINV_INTEGER_NU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tinv-integer-degrees-of-freedom",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tinv with typed-integer degrees of freedom is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TinvIntegerDegreesOfFreedomExtension"),
};
pub const TINV_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tinv-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tinv with logical numeric input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TinvLogicalInputExtension"),
};
pub const TINV_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TINV_INTEGER_P_EXTENSION,
    TINV_INTEGER_NU_EXTENSION,
    TINV_LOGICAL_INPUT_EXTENSION,
];

const INTEGER_P_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public tinv probabilities are single or double. Typed-integer probabilities are independently gated and must cross the binary64 inverse-CDF boundary exactly.",
    }];
const INTEGER_NU_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nu",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public tinv degrees of freedom are single or double. Typed-integer values are independently gated and must cross the binary64 inverse-CDF boundary exactly.",
    }];
pub const TINV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "x = tinv(integer_p, nu)",
        inputs: &INTEGER_P_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "RunMat-only integer p is checked before binary64 evaluation. Documented single input selects single output, and resident fallback restores through the exact owning provider when possible.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "x = tinv(p, integer_nu)",
        inputs: &INTEGER_NU_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "RunMat-only integer nu is checked before binary64 evaluation. Documented single input selects single output, and resident fallback restores through the exact owning provider when possible.",
    },
];
