use crate::*;

pub const HIST_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hist-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hist with integer sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HistIntegerDataExtension"),
};

pub const HIST_INTEGER_CENTERS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hist-integer-centers",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hist with integer bin centers is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HistIntegerCentersExtension"),
};

pub const HIST_MODERN_OPTIONS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hist-modern-options",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "hist with histogram-style normalization or name-value options is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HistModernOptionsExtension"),
};

pub const HIST_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    HIST_INTEGER_DATA_EXTENSION,
    HIST_INTEGER_CENTERS_EXTENSION,
    HIST_MODERN_OPTIONS_EXTENSION,
];
