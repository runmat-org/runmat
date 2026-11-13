use crate::*;
pub const TRIPULS_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tripuls-integer-sample-times",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tripuls with native typed-integer sample times is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TripulsIntegerSampleTimesExtension"),
};
pub const TRIPULS_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "tripuls-integer-controls",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "tripuls with native typed-integer width or skew is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TripulsIntegerControlsExtension"),
    };
pub const TRIPULS_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tripuls-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interactive tripuls gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TripulsGpuInputExtension"),
};
pub const TRIPULS_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TRIPULS_INTEGER_DATA_EXTENSION,
    TRIPULS_INTEGER_CONTROL_EXTENSION,
    TRIPULS_GPU_INPUT_EXTENSION,
];
