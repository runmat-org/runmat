mod documentation;

use super::support::define_metadata_predicate_entry;
use documentation::ISLOGICAL_DOCUMENTATION;

pub const ISLOGICAL_INTEGER_AUDIT: crate::BuiltinIntegerAuditDescriptor =
    crate::BuiltinIntegerAuditDescriptor {
        kind: crate::BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "islogical is a universal storage predicate; every fixed-width integer class returns false without reading or converting payload data.",
    };

define_metadata_predicate_entry!(
    entry: ISLOGICAL_CATALOG_ENTRY,
    descriptor: ISLOGICAL_DESCRIPTOR,
    internal_error: ISLOGICAL_ERROR_INTERNAL,
    output_error: ISLOGICAL_ERROR_TOO_MANY_OUTPUTS,
    name: "islogical",
    upper: "ISLOGICAL",
    predicate: crate::MetadataPredicate::Logical,
    documentation: ISLOGICAL_DOCUMENTATION,
    input_description: "Value whose logical storage class is queried.",
    output_description: "Logical scalar that is true exactly when the input uses logical storage.",
    integer_capabilities: &[],
    integer_audit: Some(&ISLOGICAL_INTEGER_AUDIT)
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISLOGICAL_CATALOG_ENTRY];
