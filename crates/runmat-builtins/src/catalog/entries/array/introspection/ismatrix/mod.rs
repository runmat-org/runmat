mod documentation;

use super::support::define_shape_predicate_entry;
use documentation::ISMATRIX_DOCUMENTATION;

define_shape_predicate_entry!(
    entry: ISMATRIX_CATALOG_ENTRY,
    descriptor: ISMATRIX_DESCRIPTOR,
    internal_error: ISMATRIX_ERROR_INTERNAL,
    output_error: ISMATRIX_ERROR_TOO_MANY_OUTPUTS,
    integer_audit: ISMATRIX_INTEGER_AUDIT,
    name: "ismatrix",
    upper: "ISMATRIX",
    predicate: crate::ShapePredicate::Matrix,
    documentation: ISMATRIX_DOCUMENTATION,
    output_description: "Logical scalar that is true when the MATLAB-visible rank is at most two.",
    integer_notes: "ismatrix is a universal shape predicate; integer class and values are irrelevant, trailing singleton dimensions are ignored, and host, resident, or distributed shape metadata is inspected without reading payload data."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISMATRIX_CATALOG_ENTRY];
