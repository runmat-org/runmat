mod documentation;

use super::support::define_shape_predicate_entry;
use documentation::ISROW_DOCUMENTATION;

define_shape_predicate_entry!(
    entry: ISROW_CATALOG_ENTRY,
    descriptor: ISROW_DESCRIPTOR,
    internal_error: ISROW_ERROR_INTERNAL,
    output_error: ISROW_ERROR_TOO_MANY_OUTPUTS,
    integer_audit: ISROW_INTEGER_AUDIT,
    name: "isrow",
    upper: "ISROW",
    predicate: crate::ShapePredicate::Row,
    documentation: ISROW_DOCUMENTATION,
    output_description: "Logical scalar that is true when the visible shape has one row.",
    integer_notes: "isrow is a universal shape predicate; integer class and values are irrelevant and host, resident, or distributed shape metadata is inspected without reading payload data."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISROW_CATALOG_ENTRY];
