mod gamrnd;

pub use gamrnd::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[gamrnd::ENTRIES];
