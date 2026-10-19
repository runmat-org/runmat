use crate::*;
pub const PULSTRAN_INTEGER_T_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pulstran-integer-time",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pulstran with typed-integer sample times is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PulstranIntegerTimeExtension"),
};
pub const PULSTRAN_INTEGER_D_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pulstran-integer-delay",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pulstran with typed-integer delays or gains is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PulstranIntegerDelayExtension"),
};
pub const PULSTRAN_INTEGER_P_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pulstran-integer-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pulstran with a typed-integer sampled prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PulstranIntegerPrototypeExtension"),
};
pub const PULSTRAN_INTEGER_PARAMETER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pulstran-integer-parameter",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pulstran with typed-integer built-in pulse parameters is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PulstranIntegerParameterExtension"),
    };
pub const PULSTRAN_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pulstran-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pulstran with an explicit gpuArray computation input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PulstranExplicitGpuInputExtension"),
    };
pub const PULSTRAN_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    PULSTRAN_INTEGER_T_EXTENSION,
    PULSTRAN_INTEGER_D_EXTENSION,
    PULSTRAN_INTEGER_P_EXTENSION,
    PULSTRAN_INTEGER_PARAMETER_EXTENSION,
    PULSTRAN_EXPLICIT_GPU_EXTENSION,
];
