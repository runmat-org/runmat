use crate::*;

pub const ICDF_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "icdf-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "icdf with typed-integer probability or parameter input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IcdfIntegerInputExtension"),
};
pub const ICDF_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "icdf-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "icdf with logical probability or parameter input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IcdfLogicalInputExtension"),
};
pub const ICDF_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [ICDF_INTEGER_INPUT_EXTENSION, ICDF_LOGICAL_INPUT_EXTENSION];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "p",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer probability input is gated before provider access and checked for exact representation at the floating inverse-CDF boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "distribution parameters",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer named-distribution parameters are gated independently from logical input and checked before floating evaluation.",
    },
];
pub const ICDF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "x = icdf(name_or_fitted_distribution, p, integer_parameters?)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "RunMat-only integer/logical inputs enter the same checked floating boundary for named and fitted-distribution-object forms. The implemented named-family subset preserves documented single/double class selection and restores resident output to the exact owner only when that owner can physically preserve the required precision; this metadata does not claim unsupported distribution families are implemented.",
    }];
