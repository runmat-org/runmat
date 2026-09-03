use crate::{BuiltinErrorDescriptor, BuiltinExtensionDescriptor, BuiltinExtensionMode};

pub const BITWISE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BITWISE.INVALID_INPUT",
    identifier: Some("RunMat:bitwise:InvalidInput"),
    when: "An input is not a supported finite integer-valued numeric, logical, or gatherable resident value.",
    message: "bitwise operation: invalid input",
};

pub const BITWISE_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BITWISE.SIZE_MISMATCH",
    identifier: Some("RunMat:bitwise:SizeMismatch"),
    when: "Input shapes violate the operation's compatible-size or scalar-or-exact-size rule.",
    message: "bitwise operation: array sizes are not compatible",
};

pub const DIRECT_BIT_GPU_UNDOCUMENTED_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "direct-bit-gpu-undocumented-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "A direct bit function with a resident input outside its documented GPU domain is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DirectBitGpuUndocumentedInputExtension"),
    };

pub const DIRECT_BIT_GPU_ASSUMED_TYPE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "direct-bit-gpu-assumedtype",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "A direct bit function with resident input and assumedtype is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DirectBitGpuAssumedTypeExtension"),
    };

pub const DIRECT_BIT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DIRECT_BIT_GPU_UNDOCUMENTED_INPUT_EXTENSION,
    DIRECT_BIT_GPU_ASSUMED_TYPE_EXTENSION,
];
