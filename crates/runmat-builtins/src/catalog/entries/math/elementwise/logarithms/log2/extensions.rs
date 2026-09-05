use crate::{BuiltinExtensionDescriptor, BuiltinExtensionMode};

pub const LOG2_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log2-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log2 with integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log2IntegerInputExtension"),
};
pub const LOG2_LOGICAL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log2-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log2 with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log2LogicalInputExtension"),
};
pub const LOG2_CHARACTER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log2-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log2 with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log2CharacterInputExtension"),
};
pub const LOG2_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    LOG2_INTEGER_EXTENSION,
    LOG2_LOGICAL_EXTENSION,
    LOG2_CHARACTER_EXTENSION,
];
