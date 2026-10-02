use crate::*;
pub const HAMMING_LOGICAL_LENGTH_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "hamming-logical-length",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "hamming with a logical scalar length is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:HammingLogicalLengthExtension"),
    };
pub const HAMMING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [HAMMING_LOGICAL_LENGTH_EXTENSION];
