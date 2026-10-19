use crate::{BuiltinExtensionDescriptor, BuiltinExtensionMode};

pub const ARRAYFUN_TEXT_CALLABLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "arrayfun-text-callable",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "arrayfun with a character-vector or string-scalar callable is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ArrayfunTextCallableExtension"),
    };
pub const ARRAYFUN_HOST_SCALAR_EXPANSION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "arrayfun-host-scalar-expansion",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "host arrayfun scalar expansion across differently sized inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ArrayfunHostScalarExpansionExtension"),
    };
pub const ARRAYFUN_GPU_OPTIONS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "arrayfun-gpu-options",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "gpuArray arrayfun with UniformOutput or ErrorHandler options is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ArrayfunGpuOptionsExtension"),
};
pub const ARRAYFUN_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    ARRAYFUN_TEXT_CALLABLE_EXTENSION,
    ARRAYFUN_HOST_SCALAR_EXPANSION_EXTENSION,
    ARRAYFUN_GPU_OPTIONS_EXTENSION,
];
