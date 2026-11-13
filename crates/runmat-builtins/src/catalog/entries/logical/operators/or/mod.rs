mod documentation;

use super::support::define_binary_logical_entry;
use documentation::OR_DOCUMENTATION;

pub const OR_COMPLEX_INPUT_EXTENSION: crate::BuiltinExtensionDescriptor =
    crate::BuiltinExtensionDescriptor {
        id: "or-complex-input",
        mode: crate::BuiltinExtensionMode::RunMatOnly,
        description: "or with a complex operand is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:OrComplexInputExtension"),
    };
pub const OR_CHARACTER_INPUT_EXTENSION: crate::BuiltinExtensionDescriptor =
    crate::BuiltinExtensionDescriptor {
        id: "or-character-input",
        mode: crate::BuiltinExtensionMode::RunMatOnly,
        description: "or with a character-array operand is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:OrCharacterInputExtension"),
    };
pub const OR_EXTENSIONS: [crate::BuiltinExtensionDescriptor; 2] =
    [OR_COMPLEX_INPUT_EXTENSION, OR_CHARACTER_INPUT_EXTENSION];

define_binary_logical_entry!(
    entry: OR_CATALOG_ENTRY,
    descriptor: OR_DESCRIPTOR,
    invalid: OR_ERROR_INVALID_INPUT,
    mismatch: OR_ERROR_SIZE_MISMATCH,
    integer_capabilities: OR_INTEGER_CAPABILITIES,
    name: "or",
    upper: "OR",
    operator: crate::LogicalBinaryOperator::Or,
    extensions: &OR_EXTENSIONS,
    documentation: OR_DOCUMENTATION,
    input_description: "Logical, real numeric, table, timetable, or supported extension operand."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&OR_CATALOG_ENTRY];
