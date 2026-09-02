mod documentation;

use super::support::define_shape_predicate_entry;
use documentation::ISSCALAR_DOCUMENTATION;

define_shape_predicate_entry!(
    entry: ISSCALAR_CATALOG_ENTRY,
    descriptor: ISSCALAR_DESCRIPTOR,
    internal_error: ISSCALAR_ERROR_INTERNAL,
    output_error: ISSCALAR_ERROR_TOO_MANY_OUTPUTS,
    integer_audit: ISSCALAR_INTEGER_AUDIT,
    name: "isscalar",
    upper: "ISSCALAR",
    predicate: crate::ShapePredicate::Scalar,
    documentation: ISSCALAR_DOCUMENTATION,
    output_description: "Logical scalar that is true when every visible dimension is one.",
    integer_notes: "isscalar is a universal shape predicate; integer class and values are irrelevant and host, resident, or distributed shape metadata is inspected without reading payload data."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISSCALAR_CATALOG_ENTRY];
