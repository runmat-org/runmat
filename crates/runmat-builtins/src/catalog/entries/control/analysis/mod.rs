mod isstable;

pub use self::isstable::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISSTABLE_CATALOG_ENTRY];
