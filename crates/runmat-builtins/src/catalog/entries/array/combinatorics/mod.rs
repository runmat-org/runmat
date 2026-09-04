mod nchoosek;
mod perms;

pub use nchoosek::*;
pub use perms::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&NCHOOSEK_CATALOG_ENTRY, &PERMS_CATALOG_ENTRY];
