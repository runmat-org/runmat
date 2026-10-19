use crate::*;

pub const INTERP2_INTEGER_SAMPLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "interp2-integer-sample",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "interp2 with typed-integer sample grids or values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Interp2IntegerSampleExtension"),
    };
pub const INTERP2_INTEGER_QUERY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "interp2-integer-query",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "interp2 with typed-integer query coordinates is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Interp2IntegerQueryExtension"),
    };
pub const INTERP2_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "interp2-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "interp2 with logical numeric input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Interp2LogicalInputExtension"),
    };
pub const INTERP2_INTEGER_EXTRAPOLATION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "interp2-integer-extrapolation",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "interp2 with a typed-integer extrapolation value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Interp2IntegerExtrapolationExtension"),
    };
pub const INTERP2_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    INTERP2_INTEGER_SAMPLE_EXTENSION,
    INTERP2_INTEGER_QUERY_EXTENSION,
    INTERP2_LOGICAL_INPUT_EXTENSION,
    INTERP2_INTEGER_EXTRAPOLATION_EXTENSION,
];
