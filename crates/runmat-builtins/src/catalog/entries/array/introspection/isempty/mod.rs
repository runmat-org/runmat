mod documentation;

use super::support::define_shape_predicate_entry;
use documentation::ISEMPTY_DOCUMENTATION;

define_shape_predicate_entry!(
    entry: ISEMPTY_CATALOG_ENTRY,
    descriptor: ISEMPTY_DESCRIPTOR,
    internal_error: ISEMPTY_ERROR_INTERNAL,
    output_error: ISEMPTY_ERROR_TOO_MANY_OUTPUTS,
    integer_audit: ISEMPTY_INTEGER_AUDIT,
    name: "isempty",
    upper: "ISEMPTY",
    predicate: crate::ShapePredicate::Empty,
    documentation: ISEMPTY_DOCUMENTATION,
    output_description: "Logical scalar that is true when the input contains zero elements.",
    integer_notes: "isempty is a universal shape predicate; integer class and values are irrelevant and host, resident, or distributed shape metadata is inspected without reading payload data."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISEMPTY_CATALOG_ENTRY];
