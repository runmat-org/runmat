mod db;
mod dcgain;
mod damp;
mod feedback;
mod isstable;
mod lqr;
mod pole;
mod pzmap;
mod rlocus;
mod zero;
#[cfg(test)]
mod tests;

pub use self::damp::*;
pub use self::db::*;
pub use self::dcgain::*;
pub use self::feedback::*;
pub use self::isstable::*;
pub use self::lqr::*;
pub use self::pole::*;
pub use self::pzmap::*;
pub use self::rlocus::*;
pub use self::zero::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &DB_CATALOG_ENTRY,
    &DCGAIN_CATALOG_ENTRY,
    &DAMP_CATALOG_ENTRY,
    &FEEDBACK_CATALOG_ENTRY,
    &ISSTABLE_CATALOG_ENTRY,
    &LQR_CATALOG_ENTRY,
    &POLE_CATALOG_ENTRY,
    &PZMAP_CATALOG_ENTRY,
    &RLOCUS_CATALOG_ENTRY,
    &ZERO_CATALOG_ENTRY,
];
