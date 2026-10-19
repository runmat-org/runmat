use crate::*;
pub const RECTPULS_INTEGER_T_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rectpuls-integer-time",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rectpuls with typed-integer sample times is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RectpulsIntegerTimeExtension"),
};
pub const RECTPULS_INTEGER_W_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rectpuls-integer-width",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rectpuls with a typed-integer width is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RectpulsIntegerWidthExtension"),
};
pub const RECTPULS_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "rectpuls-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "rectpuls with an explicit gpuArray input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RectpulsExplicitGpuInputExtension"),
    };
pub const RECTPULS_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    RECTPULS_INTEGER_T_EXTENSION,
    RECTPULS_INTEGER_W_EXTENSION,
    RECTPULS_EXPLICIT_GPU_EXTENSION,
];
