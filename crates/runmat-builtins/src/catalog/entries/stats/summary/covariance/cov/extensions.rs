use crate::*;
pub const COV_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cov-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cov with typed-integer observation data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CovIntegerDataExtension"),
};

pub const COV_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cov-logical-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cov with logical observation data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CovLogicalDataExtension"),
};

pub const COV_TYPED_NORMALIZATION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cov-typed-normalization",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cov with typed-integer or logical normalization is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CovTypedNormalizationExtension"),
    };

pub const COV_VECTOR_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cov-vector-weights",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cov with a vector of observation weights is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CovVectorWeightsExtension"),
};

pub const COV_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    COV_INTEGER_DATA_EXTENSION,
    COV_LOGICAL_DATA_EXTENSION,
    COV_TYPED_NORMALIZATION_EXTENSION,
    COV_VECTOR_WEIGHTS_EXTENSION,
];
