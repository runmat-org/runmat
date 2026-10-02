use crate::*;
pub const SAWTOOTH_NONDOUBLE_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "sawtooth-nondouble-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "sawtooth with a non-double numeric argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SawtoothNondoubleInputExtension"),
    };

pub const SAWTOOTH_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sawtooth-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interactive sawtooth gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SawtoothGpuInputExtension"),
};

pub const SAWTOOTH_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    SAWTOOTH_NONDOUBLE_INPUT_EXTENSION,
    SAWTOOTH_GPU_INPUT_EXTENSION,
];
