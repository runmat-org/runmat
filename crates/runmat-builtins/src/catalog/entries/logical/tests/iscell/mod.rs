mod documentation;

use super::support::define_metadata_predicate_entry;
use documentation::ISCELL_DOCUMENTATION;

pub const ISCELL_INTEGER_AUDIT: crate::BuiltinIntegerAuditDescriptor =
    crate::BuiltinIntegerAuditDescriptor {
        kind: crate::BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "iscell is a universal container predicate; integer values return scalar false without reading or converting payload storage.",
    };

define_metadata_predicate_entry!(
    entry: ISCELL_CATALOG_ENTRY,
    descriptor: ISCELL_DESCRIPTOR,
    internal_error: ISCELL_ERROR_INTERNAL,
    output_error: ISCELL_ERROR_TOO_MANY_OUTPUTS,
    name: "iscell",
    upper: "ISCELL",
    predicate: crate::MetadataPredicate::Cell,
    documentation: ISCELL_DOCUMENTATION,
    input_description: "Value whose container class is queried.",
    output_description: "Logical scalar that is true exactly when the input is a cell array.",
    distributed: crate::BuiltinDistributedPolicy::InspectHandles,
    integer_capabilities: &[],
    integer_audit: Some(&ISCELL_INTEGER_AUDIT)
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISCELL_CATALOG_ENTRY];
