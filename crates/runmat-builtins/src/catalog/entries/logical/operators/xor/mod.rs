mod documentation;

use super::support::define_binary_logical_entry;
use documentation::XOR_DOCUMENTATION;

pub const XOR_COMPLEX_INPUT_EXTENSION: crate::BuiltinExtensionDescriptor =
    crate::BuiltinExtensionDescriptor {
        id: "xor-complex-input",
        mode: crate::BuiltinExtensionMode::RunMatOnly,
        description: "xor with a complex operand is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:XorComplexInputExtension"),
    };
pub const XOR_EXTENSIONS: [crate::BuiltinExtensionDescriptor; 1] = [XOR_COMPLEX_INPUT_EXTENSION];

define_binary_logical_entry!(
    entry: XOR_CATALOG_ENTRY,
    descriptor: XOR_DESCRIPTOR,
    invalid: XOR_ERROR_INVALID_INPUT,
    mismatch: XOR_ERROR_SIZE_MISMATCH,
    integer_capabilities: XOR_INTEGER_CAPABILITIES,
    name: "xor",
    upper: "XOR",
    operator: crate::LogicalBinaryOperator::Xor,
    extensions: &XOR_EXTENSIONS,
    documentation: XOR_DOCUMENTATION,
    input_description: "Logical, real numeric, character, table, timetable, or supported extension operand."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&XOR_CATALOG_ENTRY];
