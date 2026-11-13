use crate::*;

pub const SPECTROGRAM_INTEGER_SIGNAL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "spectrogram-integer-signal",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "spectrogram with a typed-integer signal is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SpectrogramIntegerSignalExtension"),
    };
pub const SPECTROGRAM_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "spectrogram-integer-numeric-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "spectrogram with typed-integer numeric options is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SpectrogramIntegerNumericControlExtension"),
    };
pub const SPECTROGRAM_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    SPECTROGRAM_INTEGER_SIGNAL_EXTENSION,
    SPECTROGRAM_INTEGER_CONTROL_EXTENSION,
];
