mod isfinite;
mod isinf;
mod isnan;
mod support;

pub use isfinite::*;
pub use isinf::*;
pub use isnan::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] =
    &[isfinite::ENTRIES, isinf::ENTRIES, isnan::ENTRIES];
