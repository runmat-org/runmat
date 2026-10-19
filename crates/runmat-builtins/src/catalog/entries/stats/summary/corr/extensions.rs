use crate::*;
pub const CORR_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "corr-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "corr with typed-integer observation data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CorrIntegerDataExtension"),
};

pub const CORR_INTEGER_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "corr-integer-weights",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "corr with typed-integer observation weights is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CorrIntegerWeightsExtension"),
};

pub const CORR_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [CORR_INTEGER_DATA_EXTENSION, CORR_INTEGER_WEIGHTS_EXTENSION];
