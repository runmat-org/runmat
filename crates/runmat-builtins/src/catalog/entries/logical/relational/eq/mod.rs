mod documentation;

use super::support::define_relational_entry;
use documentation::EQ_DOCUMENTATION;

define_relational_entry!(
    entry: EQ_CATALOG_ENTRY,
    descriptor: EQ_DESCRIPTOR,
    invalid: EQ_ERROR_INVALID_INPUT,
    mismatch: EQ_ERROR_SIZE_MISMATCH,
    ownership: EQ_ERROR_PROVIDER_OWNERSHIP,
    upload: EQ_ERROR_GPU_UPLOAD,
    integer_capabilities: EQ_INTEGER_CAPABILITIES,
    name: "eq",
    upper: "EQ",
    operator: crate::RelationalOperator::Equal,
    documentation: EQ_DOCUMENTATION,
    output_description: "Logical equality result, or a symbolic equation for symbolic operands."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&EQ_CATALOG_ENTRY];
