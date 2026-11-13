use crate::*;
pub const BLACKMAN_LOGICAL_LENGTH_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "blackman-logical-length",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "blackman with a logical scalar length is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BlackmanLogicalLengthExtension"),
    };
pub const BLACKMAN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [BLACKMAN_LOGICAL_LENGTH_EXTENSION];
