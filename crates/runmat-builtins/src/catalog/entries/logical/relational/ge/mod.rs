mod documentation;

use super::support::define_relational_entry;
use documentation::GE_DOCUMENTATION;

define_relational_entry!(
    entry: GE_CATALOG_ENTRY,
    descriptor: GE_DESCRIPTOR,
    invalid: GE_ERROR_INVALID_INPUT,
    mismatch: GE_ERROR_SIZE_MISMATCH,
    ownership: GE_ERROR_PROVIDER_OWNERSHIP,
    upload: GE_ERROR_GPU_UPLOAD,
    integer_capabilities: GE_INTEGER_CAPABILITIES,
    name: "ge",
    upper: "GE",
    operator: crate::RelationalOperator::GreaterThanOrEqual,
    documentation: GE_DOCUMENTATION,
    output_description: "Logical greater-than-or-equal result, or a symbolic relation for symbolic operands."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&GE_CATALOG_ENTRY];
