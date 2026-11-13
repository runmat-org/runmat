use crate::*;
pub const CORRCOEF_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "corrcoef-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "corrcoef with typed-integer observation data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CorrcoefIntegerDataExtension"),
    };

pub const CORRCOEF_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "corrcoef-logical-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "corrcoef with logical observation data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CorrcoefLogicalDataExtension"),
    };

pub const CORRCOEF_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    CORRCOEF_INTEGER_DATA_EXTENSION,
    CORRCOEF_LOGICAL_DATA_EXTENSION,
];
