use crate::*;
pub const FILTFILT_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "filtfilt-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "filtfilt with typed-integer coefficients or signal data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FiltfiltIntegerInputExtension"),
    };

pub const FILTFILT_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "filtfilt-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "filtfilt with logical coefficients or signal data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FiltfiltLogicalInputExtension"),
    };

pub const FILTFILT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    FILTFILT_INTEGER_INPUT_EXTENSION,
    FILTFILT_LOGICAL_INPUT_EXTENSION,
];
