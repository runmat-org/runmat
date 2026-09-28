mod db;
mod dcgain;
mod damp;
mod isstable;

pub use self::damp::*;
pub use self::db::*;
pub use self::dcgain::*;
pub use self::isstable::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &DB_CATALOG_ENTRY,
    &DCGAIN_CATALOG_ENTRY,
    &DAMP_CATALOG_ENTRY,
    &ISSTABLE_CATALOG_ENTRY,
];
