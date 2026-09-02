mod binornd;
mod gamrnd;

pub use binornd::*;
pub use gamrnd::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] =
    &[binornd::ENTRIES, gamrnd::ENTRIES];
