use crate::*;

pub const HISTCOUNTS2_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "histcounts2-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "histcounts2 with gpuArray inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Histcounts2GpuInputExtension"),
};
pub const HISTCOUNTS2_EXTRA_SYNTAX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "histcounts2-extra-bin-syntax",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "histcounts2 axis-specific or scalar-width convenience syntax is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Histcounts2ExtraBinSyntaxExtension"),
    };
pub const HISTCOUNTS2_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    HISTCOUNTS2_GPU_EXTENSION,
    HISTCOUNTS2_EXTRA_SYNTAX_EXTENSION,
];
