use crate::*;

pub const STACKEDPLOT_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "stackedplot-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "stackedplot with an explicitly GPU-resident input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StackedplotExplicitGpuInputExtension"),
    };

pub const STACKEDPLOT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [STACKEDPLOT_EXPLICIT_GPU_EXTENSION];
