mod documentation;

use super::support::define_relational_entry;
use documentation::GT_DOCUMENTATION;

define_relational_entry!(
    entry: GT_CATALOG_ENTRY,
    descriptor: GT_DESCRIPTOR,
    invalid: GT_ERROR_INVALID_INPUT,
    mismatch: GT_ERROR_SIZE_MISMATCH,
    ownership: GT_ERROR_PROVIDER_OWNERSHIP,
    upload: GT_ERROR_GPU_UPLOAD,
    integer_capabilities: GT_INTEGER_CAPABILITIES,
    name: "gt",
    upper: "GT",
    operator: crate::RelationalOperator::GreaterThan,
    documentation: GT_DOCUMENTATION,
    output_description: "Logical greater-than result, or a symbolic relation for symbolic operands."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&GT_CATALOG_ENTRY];
