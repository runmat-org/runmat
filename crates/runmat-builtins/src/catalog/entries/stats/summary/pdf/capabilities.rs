use crate::*;

pub const PDF_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pdf with typed-integer evaluation points is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PdfIntegerXExtension"),
};
pub const PDF_INTEGER_PARAMETER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pdf-integer-parameters",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pdf with typed-integer distribution parameters is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PdfIntegerParametersExtension"),
    };
pub const PDF_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [PDF_INTEGER_X_EXTENSION, PDF_INTEGER_PARAMETER_EXTENSION];
const PDF_INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double evaluation points. RunMat gates typed integers before provider access and requires exact conversion at the floating density boundary.",
    }];
const PDF_INTEGER_PARAMETER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A...D",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double named-distribution parameters. RunMat independently gates typed parameters and rejects lossy wide values.",
    }];
pub const PDF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = pdf(pd, integer_x) or pdf(name, integer_x, A...)",
        inputs: &PDF_INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed evaluation points are a RunMat-only floating-boundary extension; density or mass values are floating outputs, and resident fallback restores the selected floating output class through the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = pdf(name, x, integer_A...D)",
        inputs: &PDF_INTEGER_PARAMETER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed distribution parameters are independently gated and converted only after exactness is proved.",
    },
];
