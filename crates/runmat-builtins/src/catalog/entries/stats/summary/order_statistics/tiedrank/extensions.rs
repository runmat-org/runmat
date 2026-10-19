use crate::*;

pub const DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tiedrank-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tiedrank with typed-integer input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TiedrankIntegerDataExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [DATA_EXTENSION];
