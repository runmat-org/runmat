use crate::*;
pub const CORRCOV_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "corrcov-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "corrcov with typed-integer covariance data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CorrcovIntegerDataExtension"),
};

pub const CORRCOV_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "corrcov-logical-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "corrcov with logical covariance data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CorrcovLogicalDataExtension"),
};

pub const CORRCOV_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    CORRCOV_INTEGER_DATA_EXTENSION,
    CORRCOV_LOGICAL_DATA_EXTENSION,
];

pub const COV2CORR_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cov2corr-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cov2corr with typed-integer covariance data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Cov2corrIntegerDataExtension"),
    };

pub const COV2CORR_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cov2corr-logical-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cov2corr with logical covariance data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Cov2corrLogicalDataExtension"),
    };

pub const COV2CORR_SINGLE_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cov2corr-single-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cov2corr with single covariance data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Cov2corrSingleDataExtension"),
};

pub const COV2CORR_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cov2corr-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cov2corr with a resident GPU covariance matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Cov2corrGpuInputExtension"),
};

pub const COV2CORR_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    COV2CORR_INTEGER_DATA_EXTENSION,
    COV2CORR_LOGICAL_DATA_EXTENSION,
    COV2CORR_SINGLE_DATA_EXTENSION,
    COV2CORR_GPU_INPUT_EXTENSION,
];
