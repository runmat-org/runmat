mod documentation;

use super::support::define_metadata_predicate_entry;
use documentation::ISCELLSTR_DOCUMENTATION;

pub const ISCELLSTR_INTEGER_AUDIT: crate::BuiltinIntegerAuditDescriptor =
    crate::BuiltinIntegerAuditDescriptor {
        kind: crate::BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "iscellstr is a universal container-content predicate; integer values or integer cell members return scalar false without numeric conversion.",
    };

define_metadata_predicate_entry!(
    entry: ISCELLSTR_CATALOG_ENTRY,
    descriptor: ISCELLSTR_DESCRIPTOR,
    internal_error: ISCELLSTR_ERROR_INTERNAL,
    output_error: ISCELLSTR_ERROR_TOO_MANY_OUTPUTS,
    name: "iscellstr",
    upper: "ISCELLSTR",
    predicate: crate::MetadataPredicate::CellString,
    documentation: ISCELLSTR_DOCUMENTATION,
    input_description: "Value tested for cell-array-of-character-arrays content.",
    output_description: "Logical scalar that is true for an empty cell array or a cell array containing only character arrays.",
    distributed: crate::BuiltinDistributedPolicy::MaterializeArguments,
    integer_capabilities: &[],
    integer_audit: Some(&ISCELLSTR_INTEGER_AUDIT)
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISCELLSTR_CATALOG_ENTRY];
