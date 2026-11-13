use crate::{BuiltinExtensionDescriptor, BuiltinExtensionMode};

pub const BUTTORD_INTEGER_FREQUENCY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "buttord-integer-frequency",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "buttord with typed-integer passband or stopband edges is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ButtordIntegerFrequencyExtension"),
    };
pub const BUTTORD_LOGICAL_FREQUENCY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "buttord-logical-frequency",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "buttord with logical passband or stopband edges is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ButtordLogicalFrequencyExtension"),
    };
pub const BUTTORD_COMPLEX_FREQUENCY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "buttord-complex-frequency",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "buttord with complex passband or stopband edges is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ButtordComplexFrequencyExtension"),
    };
pub const BUTTORD_INTEGER_ATTENUATION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "buttord-integer-attenuation",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "buttord with typed-integer ripple or attenuation is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ButtordIntegerAttenuationExtension"),
    };
pub const BUTTORD_LOGICAL_ATTENUATION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "buttord-logical-attenuation",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "buttord with logical ripple or attenuation is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ButtordLogicalAttenuationExtension"),
    };
pub const BUTTORD_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "buttord-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "buttord with resident GPU-array design arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ButtordGpuInputExtension"),
};

pub const BUTTORD_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    BUTTORD_INTEGER_FREQUENCY_EXTENSION,
    BUTTORD_LOGICAL_FREQUENCY_EXTENSION,
    BUTTORD_COMPLEX_FREQUENCY_EXTENSION,
    BUTTORD_INTEGER_ATTENUATION_EXTENSION,
    BUTTORD_LOGICAL_ATTENUATION_EXTENSION,
    BUTTORD_GPU_INPUT_EXTENSION,
];
