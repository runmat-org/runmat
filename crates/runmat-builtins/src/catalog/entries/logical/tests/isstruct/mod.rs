mod documentation;

use super::support::define_metadata_predicate_entry;
use documentation::ISSTRUCT_DOCUMENTATION;

pub const ISSTRUCT_INTEGER_AUDIT: crate::BuiltinIntegerAuditDescriptor =
    crate::BuiltinIntegerAuditDescriptor {
        kind: crate::BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "isstruct is a universal container predicate; integer inputs return scalar false without payload conversion or inspection.",
    };

define_metadata_predicate_entry!(
    entry: ISSTRUCT_CATALOG_ENTRY,
    descriptor: ISSTRUCT_DESCRIPTOR,
    internal_error: ISSTRUCT_ERROR_INTERNAL,
    output_error: ISSTRUCT_ERROR_TOO_MANY_OUTPUTS,
    name: "isstruct",
    upper: "ISSTRUCT",
    predicate: crate::MetadataPredicate::Struct,
    documentation: ISSTRUCT_DOCUMENTATION,
    input_description: "Value whose structure-container class is queried.",
    output_description: "Logical scalar that is true exactly when the input is a structure array.",
    distributed: crate::BuiltinDistributedPolicy::InspectHandles,
    integer_capabilities: &[],
    integer_audit: Some(&ISSTRUCT_INTEGER_AUDIT)
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISSTRUCT_CATALOG_ENTRY];
