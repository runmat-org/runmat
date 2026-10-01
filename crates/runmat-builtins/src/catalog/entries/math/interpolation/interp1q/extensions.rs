use crate::*;

pub const INTERP1Q_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "interp1q-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "interp1q with typed-integer X or V data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Interp1qIntegerDataExtension"),
    };
pub const INTERP1Q_INTEGER_QUERY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "interp1q-integer-query",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "interp1q with typed-integer query coordinates is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Interp1qIntegerQueryExtension"),
    };
pub const INTERP1Q_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "interp1q-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "interp1q with logical numeric input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Interp1qLogicalInputExtension"),
    };
pub const INTERP1Q_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "interp1q-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interp1q with resident gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Interp1qGpuInputExtension"),
};
pub const INTERP1Q_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    INTERP1Q_INTEGER_DATA_EXTENSION,
    INTERP1Q_INTEGER_QUERY_EXTENSION,
    INTERP1Q_LOGICAL_INPUT_EXTENSION,
    INTERP1Q_GPU_INPUT_EXTENSION,
];
