mod documentation;

use super::support::define_relational_entry;
use documentation::NE_DOCUMENTATION;

define_relational_entry!(
    entry: NE_CATALOG_ENTRY,
    descriptor: NE_DESCRIPTOR,
    invalid: NE_ERROR_INVALID_INPUT,
    mismatch: NE_ERROR_SIZE_MISMATCH,
    ownership: NE_ERROR_PROVIDER_OWNERSHIP,
    upload: NE_ERROR_GPU_UPLOAD,
    integer_capabilities: NE_INTEGER_CAPABILITIES,
    name: "ne",
    upper: "NE",
    operator: crate::RelationalOperator::NotEqual,
    documentation: NE_DOCUMENTATION,
    output_description: "Logical inequality result, or a symbolic inequality for symbolic operands."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&NE_CATALOG_ENTRY];
