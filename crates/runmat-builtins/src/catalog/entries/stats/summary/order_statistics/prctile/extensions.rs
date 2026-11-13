use crate::*;

pub const DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "prctile-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "prctile with typed-integer input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PrctileIntegerDataExtension"),
};
pub const PROBABILITY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "prctile-typed-integer-percentage",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "prctile with typed-integer percentages is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PrctileTypedIntegerPercentageExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [DATA_EXTENSION, PROBABILITY_EXTENSION];
