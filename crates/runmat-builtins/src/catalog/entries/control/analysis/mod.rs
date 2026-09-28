mod damp;
mod isstable;

pub use self::damp::*;
pub use self::isstable::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&DAMP_CATALOG_ENTRY, &ISSTABLE_CATALOG_ENTRY];
