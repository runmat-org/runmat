use crate::*;

pub const DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "quantile-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "quantile with typed-integer input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QuantileIntegerDataExtension"),
};
pub const PROBABILITY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "quantile-typed-integer-probability",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "quantile with typed-integer probabilities is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QuantileTypedIntegerProbabilityExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [DATA_EXTENSION, PROBABILITY_EXTENSION];
