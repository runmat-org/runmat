mod binornd;
mod gamrnd;
mod wblrnd;

pub use binornd::*;
pub use gamrnd::*;
pub use wblrnd::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] =
    &[binornd::ENTRIES, gamrnd::ENTRIES, wblrnd::ENTRIES];
