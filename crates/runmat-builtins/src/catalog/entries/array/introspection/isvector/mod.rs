mod documentation;

use super::support::define_shape_predicate_entry;
use documentation::ISVECTOR_DOCUMENTATION;

define_shape_predicate_entry!(
    entry: ISVECTOR_CATALOG_ENTRY,
    descriptor: ISVECTOR_DESCRIPTOR,
    internal_error: ISVECTOR_ERROR_INTERNAL,
    output_error: ISVECTOR_ERROR_TOO_MANY_OUTPUTS,
    integer_audit: ISVECTOR_INTEGER_AUDIT,
    name: "isvector",
    upper: "ISVECTOR",
    predicate: crate::ShapePredicate::Vector,
    documentation: ISVECTOR_DOCUMENTATION,
    output_description: "Logical scalar that is true for a visible 1-by-N or N-by-1 shape.",
    integer_notes: "isvector is a universal shape predicate; integer class and values are irrelevant, trailing singleton dimensions are ignored, and host, resident, or distributed shape metadata is inspected without reading payload data."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISVECTOR_CATALOG_ENTRY];
