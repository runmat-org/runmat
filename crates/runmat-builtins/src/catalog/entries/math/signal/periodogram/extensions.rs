use crate::*;

pub const PERIODOGRAM_INTEGER_SIGNAL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "periodogram-integer-signal",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "periodogram accepts a typed-integer input signal as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PeriodogramIntegerSignalExtension"),
    };
pub const PERIODOGRAM_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "periodogram-integer-numeric-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "periodogram accepts typed-integer window, frequency, DFT-length, or sample-rate controls as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PeriodogramIntegerNumericControlExtension"),
    };
pub const PERIODOGRAM_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    PERIODOGRAM_INTEGER_SIGNAL_EXTENSION,
    PERIODOGRAM_INTEGER_CONTROL_EXTENSION,
];
