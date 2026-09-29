//! Dynamic-system response operations.

mod impulse;
mod step;

mod inference;

pub use self::impulse::*;
pub use self::step::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&IMPULSE_CATALOG_ENTRY, &STEP_CATALOG_ENTRY];
