use crate::*;
pub const HISTC_WIDE_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "histc-wide-integer",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "histc with int64 or uint64 input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HistcWideIntegerExtension"),
};
pub const HISTC_LOGICAL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "histc-logical",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "histc with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HistcLogicalExtension"),
};
pub const HISTC_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "histc-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "histc with gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HistcGpuInputExtension"),
};
pub const HISTC_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    HISTC_WIDE_INTEGER_EXTENSION,
    HISTC_LOGICAL_EXTENSION,
    HISTC_GPU_EXTENSION,
];
