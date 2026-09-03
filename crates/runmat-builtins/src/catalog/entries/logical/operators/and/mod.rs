mod documentation;

use super::support::define_binary_logical_entry;
use documentation::AND_DOCUMENTATION;

pub const AND_COMPLEX_INPUT_EXTENSION: crate::BuiltinExtensionDescriptor =
    crate::BuiltinExtensionDescriptor {
        id: "and-complex-input",
        mode: crate::BuiltinExtensionMode::RunMatOnly,
        description: "and with a complex operand is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AndComplexInputExtension"),
    };
pub const AND_CHARACTER_INPUT_EXTENSION: crate::BuiltinExtensionDescriptor =
    crate::BuiltinExtensionDescriptor {
        id: "and-character-input",
        mode: crate::BuiltinExtensionMode::RunMatOnly,
        description: "and with a character-array operand is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AndCharacterInputExtension"),
    };
pub const AND_EXTENSIONS: [crate::BuiltinExtensionDescriptor; 2] =
    [AND_COMPLEX_INPUT_EXTENSION, AND_CHARACTER_INPUT_EXTENSION];

define_binary_logical_entry!(
    entry: AND_CATALOG_ENTRY,
    descriptor: AND_DESCRIPTOR,
    invalid: AND_ERROR_INVALID_INPUT,
    mismatch: AND_ERROR_SIZE_MISMATCH,
    integer_capabilities: AND_INTEGER_CAPABILITIES,
    name: "and",
    upper: "AND",
    operator: crate::LogicalBinaryOperator::And,
    extensions: &AND_EXTENSIONS,
    documentation: AND_DOCUMENTATION,
    input_description: "Logical, real numeric, table, timetable, or supported extension operand."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&AND_CATALOG_ENTRY];
