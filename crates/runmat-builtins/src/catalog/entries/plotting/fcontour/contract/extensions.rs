use crate::*;

pub const FCONTOUR_INTEGER_LINE_COLOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fcontour-integer-line-color",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fcontour with a typed-integer RGB LineColor is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FcontourIntegerLineColorExtension"),
    };

pub const FCONTOUR_POSITIONAL_LEVEL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fcontour-positional-level-spec",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fcontour with a positional level count or value vector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FcontourPositionalLevelSpecExtension"),
    };

pub const FCONTOUR_RESIDENT_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fcontour-resident-numeric-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fcontour with resident numeric arguments is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FcontourResidentNumericInputExtension"),
    };

pub const FCONTOUR_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FCONTOUR_INTEGER_LINE_COLOR_EXTENSION,
    FCONTOUR_POSITIONAL_LEVEL_EXTENSION,
    FCONTOUR_RESIDENT_NUMERIC_EXTENSION,
];
