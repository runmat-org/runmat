use crate::*;

pub const PWELCH_INTEGER_SIGNAL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pwelch-integer-signal",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pwelch accepts a typed-integer input signal as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PwelchIntegerSignalExtension"),
    };
pub const PWELCH_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pwelch-integer-numeric-control", mode: BuiltinExtensionMode::RunMatOnly,
    description: "pwelch accepts typed-integer window, overlap, frequency, DFT-length, or sample-rate controls as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PwelchIntegerNumericControlExtension"),
};
pub const PWELCH_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pwelch-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pwelch accepts logical signal or numeric-control input as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PwelchLogicalInputExtension"),
};
pub const PWELCH_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    PWELCH_INTEGER_SIGNAL_EXTENSION,
    PWELCH_INTEGER_CONTROL_EXTENSION,
    PWELCH_LOGICAL_INPUT_EXTENSION,
];
