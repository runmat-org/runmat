mod combinations;
mod nchoosek;
mod perms;

pub use combinations::*;
pub use nchoosek::*;
pub use perms::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &COMBINATIONS_CATALOG_ENTRY,
    &NCHOOSEK_CATALOG_ENTRY,
    &PERMS_CATALOG_ENTRY,
];
