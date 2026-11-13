use crate::*;

pub const TRIPLOT_INTEGER_CONNECTIVITY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "triplot-integer-connectivity",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "triplot with native typed-integer connectivity is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TriplotIntegerConnectivityExtension"),
    };

pub const TRIPLOT_INTEGER_COORDINATE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "triplot-integer-coordinates",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "triplot with native typed-integer coordinates is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TriplotIntegerCoordinatesExtension"),
    };

pub const TRIPLOT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "triplot-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interactive triplot gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TriplotGpuInputExtension"),
};

pub const TRIPLOT_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TRIPLOT_INTEGER_CONNECTIVITY_EXTENSION,
    TRIPLOT_INTEGER_COORDINATE_EXTENSION,
    TRIPLOT_GPU_INPUT_EXTENSION,
];
