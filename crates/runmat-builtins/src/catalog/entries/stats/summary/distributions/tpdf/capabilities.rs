use crate::*;

pub const TPDF_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tpdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tpdf accepts typed-integer evaluation points",
    error_identifier: Some("RunMat:compatibility:TpdfIntegerXExtension"),
};
pub const TPDF_INTEGER_NU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tpdf-integer-degrees-of-freedom",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tpdf accepts typed-integer degrees of freedom",
    error_identifier: Some("RunMat:compatibility:TpdfIntegerDegreesOfFreedomExtension"),
};
pub const TPDF_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tpdf-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tpdf accepts logical numeric input",
    error_identifier: Some("RunMat:compatibility:TpdfLogicalInputExtension"),
};
pub const TPDF_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TPDF_INTEGER_X_EXTENSION,
    TPDF_INTEGER_NU_EXTENSION,
    TPDF_LOGICAL_INPUT_EXTENSION,
];

const INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public tpdf evaluation points are single or double; native integers are independently gated and must cross the binary64 PDF boundary exactly.",
    }];
const INTEGER_NU_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nu",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public tpdf degrees of freedom are single or double; native integers are independently gated and must cross the binary64 PDF boundary exactly.",
    }];
pub const TPDF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "y = tpdf(integer_x, nu)", inputs: &INTEGER_X_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar, notes: "Admitted integer x values are checked before binary64 evaluation. Documented single input selects single output, and resident fallback restores through the exact owning provider." },
    BuiltinIntegerCapabilityDescriptor { form: "y = tpdf(x, integer_nu)", inputs: &INTEGER_NU_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar, notes: "Admitted integer nu values are checked before binary64 evaluation. Documented single input selects single output, and resident fallback restores through the exact owning provider." },
];
