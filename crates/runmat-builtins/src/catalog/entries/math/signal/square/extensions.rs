use crate::*;
pub const SQUARE_NONFLOATING_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "square-nonfloating-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "square with an integer or logical argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SquareNonfloatingInputExtension"),
    };

pub const SQUARE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "square-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interactive square gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SquareGpuInputExtension"),
};

pub const SQUARE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    SQUARE_NONFLOATING_INPUT_EXTENSION,
    SQUARE_GPU_INPUT_EXTENSION,
];
