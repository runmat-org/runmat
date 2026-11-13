mod documentation;

use super::support::define_relational_entry;
use documentation::LT_DOCUMENTATION;

define_relational_entry!(
    entry: LT_CATALOG_ENTRY,
    descriptor: LT_DESCRIPTOR,
    invalid: LT_ERROR_INVALID_INPUT,
    mismatch: LT_ERROR_SIZE_MISMATCH,
    ownership: LT_ERROR_PROVIDER_OWNERSHIP,
    upload: LT_ERROR_GPU_UPLOAD,
    integer_capabilities: LT_INTEGER_CAPABILITIES,
    name: "lt",
    upper: "LT",
    operator: crate::RelationalOperator::LessThan,
    documentation: LT_DOCUMENTATION,
    output_description: "Logical less-than result, or a symbolic relation for symbolic operands."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&LT_CATALOG_ENTRY];
