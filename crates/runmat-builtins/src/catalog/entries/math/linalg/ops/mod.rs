mod ctranspose;

pub use ctranspose::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&CTRANSPOSE_CATALOG_ENTRY];

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(ENTRIES.iter().copied());
}
