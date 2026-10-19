mod documentation;

use super::support::define_shape_predicate_entry;
use documentation::ISCOLUMN_DOCUMENTATION;

define_shape_predicate_entry!(
    entry: ISCOLUMN_CATALOG_ENTRY,
    descriptor: ISCOLUMN_DESCRIPTOR,
    internal_error: ISCOLUMN_ERROR_INTERNAL,
    output_error: ISCOLUMN_ERROR_TOO_MANY_OUTPUTS,
    integer_audit: ISCOLUMN_INTEGER_AUDIT,
    name: "iscolumn",
    upper: "ISCOLUMN",
    predicate: crate::ShapePredicate::Column,
    documentation: ISCOLUMN_DOCUMENTATION,
    output_description: "Logical scalar that is true when the visible shape has one column.",
    integer_notes: "iscolumn is a universal shape predicate; integer class and values are irrelevant and host, resident, or distributed shape metadata is inspected without reading payload data."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISCOLUMN_CATALOG_ENTRY];
