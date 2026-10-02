use crate::*;
pub const HANN_LOGICAL_LENGTH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hann-logical-length",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hann with a logical scalar length is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HannLogicalLengthExtension"),
};
pub const HANN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [HANN_LOGICAL_LENGTH_EXTENSION];
