mod ceil;
mod fix;
mod floor;
mod modulus;
mod rem;
mod round;

pub use ceil::*;
pub use fix::*;
pub use floor::*;
pub use modulus::*;
pub use rem::*;
pub use round::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CEIL_CATALOG_ENTRY,
    &FIX_CATALOG_ENTRY,
    &FLOOR_CATALOG_ENTRY,
    &MOD_CATALOG_ENTRY,
    &REM_CATALOG_ENTRY,
    &ROUND_CATALOG_ENTRY,
];
