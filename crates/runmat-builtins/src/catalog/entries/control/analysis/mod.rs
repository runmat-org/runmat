mod db;
mod dcgain;
mod damp;
mod feedback;
mod isstable;
mod lqr;

pub use self::damp::*;
pub use self::db::*;
pub use self::dcgain::*;
pub use self::feedback::*;
pub use self::isstable::*;
pub use self::lqr::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &DB_CATALOG_ENTRY,
    &DCGAIN_CATALOG_ENTRY,
    &DAMP_CATALOG_ENTRY,
    &FEEDBACK_CATALOG_ENTRY,
    &ISSTABLE_CATALOG_ENTRY,
    &LQR_CATALOG_ENTRY,
];
