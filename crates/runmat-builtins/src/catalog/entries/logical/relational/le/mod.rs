mod documentation;

use super::support::define_relational_entry;
use documentation::LE_DOCUMENTATION;

define_relational_entry!(
    entry: LE_CATALOG_ENTRY,
    descriptor: LE_DESCRIPTOR,
    invalid: LE_ERROR_INVALID_INPUT,
    mismatch: LE_ERROR_SIZE_MISMATCH,
    ownership: LE_ERROR_PROVIDER_OWNERSHIP,
    upload: LE_ERROR_GPU_UPLOAD,
    integer_capabilities: LE_INTEGER_CAPABILITIES,
    name: "le",
    upper: "LE",
    operator: crate::RelationalOperator::LessThanOrEqual,
    documentation: LE_DOCUMENTATION,
    output_description: "Logical less-than-or-equal result, or a symbolic relation for symbolic operands."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&LE_CATALOG_ENTRY];
