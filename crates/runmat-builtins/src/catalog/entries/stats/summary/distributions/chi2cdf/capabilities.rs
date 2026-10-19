use crate::*;

pub const CHI2CDF_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "chi2cdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "chi2cdf with typed-integer evaluation points is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Chi2cdfIntegerXExtension"),
};
pub const CHI2CDF_INTEGER_NU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "chi2cdf-integer-degrees-of-freedom",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "chi2cdf with typed-integer degrees of freedom is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Chi2cdfIntegerDegreesOfFreedomExtension"),
};
pub const CHI2CDF_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "chi2cdf-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "chi2cdf with logical numeric input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Chi2cdfLogicalInputExtension"),
    };
pub const CHI2CDF_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    CHI2CDF_INTEGER_X_EXTENSION,
    CHI2CDF_INTEGER_NU_EXTENSION,
    CHI2CDF_LOGICAL_INPUT_EXTENSION,
];

const INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer evaluation points are outside the documented single/double input domain and cross a checked floating boundary only in RunMat mode.",
    }];
const INTEGER_NU_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nu",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer degrees of freedom are outside the documented single/double input domain and cross a checked floating boundary only in RunMat mode.",
    }];
pub const CHI2CDF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "p = chi2cdf(integer_x, nu, ...)",
        inputs: &INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "RunMat-only integer x is checked for exact binary64 representation before CDF evaluation; documented single input selects single output and documented resident calls return to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "p = chi2cdf(x, integer_nu, ...)",
        inputs: &INTEGER_NU_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "RunMat-only integer nu is checked for exact binary64 representation before CDF evaluation; documented single input selects single output and documented resident calls return to the owning provider.",
    },
];
