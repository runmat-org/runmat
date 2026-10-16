mod ctranspose;
mod inference;
mod cross;
mod dot;
mod pagemtimes;
mod pagetranspose;
mod trace;
mod transpose;

pub use cross::*;
pub use ctranspose::*;
pub use dot::*;
pub use pagemtimes::*;
pub use pagetranspose::*;
pub use trace::*;
pub use transpose::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CTRANSPOSE_CATALOG_ENTRY,
    &CROSS_CATALOG_ENTRY,
    &DOT_CATALOG_ENTRY,
    &PAGEMTIMES_CATALOG_ENTRY,
    &PAGETRANSPOSE_CATALOG_ENTRY,
    &TRACE_CATALOG_ENTRY,
    &TRANSPOSE_CATALOG_ENTRY,
];

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(ENTRIES.iter().copied());
}
