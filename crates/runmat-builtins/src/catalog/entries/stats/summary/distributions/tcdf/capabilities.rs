use crate::*;

pub const TCDF_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcdf with typed-integer evaluation points is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcdfIntegerXExtension"),
};
pub const TCDF_INTEGER_NU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcdf-integer-degrees-of-freedom",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcdf with typed-integer degrees of freedom is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcdfIntegerDegreesOfFreedomExtension"),
};
pub const TCDF_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcdf-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcdf with logical numeric input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcdfLogicalInputExtension"),
};
pub const TCDF_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TCDF_INTEGER_X_EXTENSION,
    TCDF_INTEGER_NU_EXTENSION,
    TCDF_LOGICAL_INPUT_EXTENSION,
];

const INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public tcdf inputs are single or double. Typed-integer evaluation points are independently gated and must cross the binary64 CDF boundary exactly.",
    }];
const INTEGER_NU_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nu",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public tcdf degrees of freedom are single or double. Typed-integer values are independently gated and must cross the binary64 CDF boundary exactly.",
    }];
pub const TCDF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "p = tcdf(integer_x, nu [, \"upper\"])",
        inputs: &INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "RunMat-only integer x is checked before binary64 evaluation. Documented single input selects single output, and resident fallback restores through the exact owning provider when possible.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "p = tcdf(x, integer_nu [, \"upper\"])",
        inputs: &INTEGER_NU_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "RunMat-only integer nu is checked before binary64 evaluation. Documented single input selects single output, and resident fallback restores through the exact owning provider when possible.",
    },
];
