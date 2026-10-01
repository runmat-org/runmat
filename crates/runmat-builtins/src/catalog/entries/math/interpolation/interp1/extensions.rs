use crate::*;

pub const INTERP1_INTEGER_SAMPLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "interp1-integer-sample",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "interp1 with typed-integer sample locations or values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Interp1IntegerSampleExtension"),
    };
pub const INTERP1_INTEGER_QUERY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "interp1-integer-query",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "interp1 with typed-integer query coordinates is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Interp1IntegerQueryExtension"),
    };
pub const INTERP1_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "interp1-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "interp1 with logical numeric input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Interp1LogicalInputExtension"),
    };
pub const INTERP1_INTEGER_EXTRAPOLATION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "interp1-integer-extrapolation",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "interp1 with a typed-integer extrapolation value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Interp1IntegerExtrapolationExtension"),
    };
pub const INTERP1_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    INTERP1_INTEGER_SAMPLE_EXTENSION,
    INTERP1_INTEGER_QUERY_EXTENSION,
    INTERP1_LOGICAL_INPUT_EXTENSION,
    INTERP1_INTEGER_EXTRAPOLATION_EXTENSION,
];
