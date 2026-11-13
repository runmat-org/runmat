//! DataArray receiver identities.

mod metadata;
mod payload;
#[cfg(test)]
mod tests;

pub use metadata::*;
pub use payload::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &DATAARRAY_CHUNK_SHAPE_CATALOG_ENTRY,
    &DATAARRAY_CODEC_CATALOG_ENTRY,
    &DATAARRAY_DTYPE_CATALOG_ENTRY,
    &DATAARRAY_FILL_CATALOG_ENTRY,
    &DATAARRAY_NAME_CATALOG_ENTRY,
    &DATAARRAY_RANK_CATALOG_ENTRY,
    &DATAARRAY_READ_CATALOG_ENTRY,
    &DATAARRAY_RESIZE_CATALOG_ENTRY,
    &DATAARRAY_SHAPE_CATALOG_ENTRY,
    &DATAARRAY_WRITE_CATALOG_ENTRY,
];
