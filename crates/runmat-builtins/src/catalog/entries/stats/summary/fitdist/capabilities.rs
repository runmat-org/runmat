use crate::*;

pub const FITDIST_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fitdist-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fitdist with typed-integer sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FitdistIntegerDataExtension"),
};
pub const FITDIST_INTEGER_FREQUENCY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitdist-integer-frequency",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitdist with typed-integer Frequency storage is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FitdistIntegerFrequencyExtension"),
    };
pub const FITDIST_RESIDENT_FALLBACK_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitdist-resident-fallback",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitdist gathers resident inputs and returns a host distribution object",
        error_identifier: Some("RunMat:compatibility:FitdistResidentFallbackExtension"),
    };
pub const FITDIST_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FITDIST_INTEGER_DATA_EXTENSION,
    FITDIST_INTEGER_FREQUENCY_EXTENSION,
    FITDIST_RESIDENT_FALLBACK_EXTENSION,
];
const FITDIST_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents double sample data. RunMat mode admits all eight integer classes only when every observation is exactly representable at the binary64 fitting boundary.",
    }];
const FITDIST_INTEGER_FREQUENCY_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Frequency",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target requires nonnegative integer-valued single or double counts. RunMat mode additionally admits exact typed-integer count storage.",
    }];
pub const FITDIST_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "pd = fitdist(integer_x, distname, ...)",
        inputs: &FITDIST_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer observations are gated before provider access, gathered exactly when resident, and converted once to binary64; the returned host probability-distribution object has floating parameters.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "pd = fitdist(x, distname, Frequency=integer_counts)",
        inputs: &FITDIST_INTEGER_FREQUENCY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Typed counts are independently gated before provider access and must remain nonnegative, finite, integral, shape-matched, and exactly representable as binary64 weights.",
    },
];
