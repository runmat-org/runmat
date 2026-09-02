mod all;
mod any;
mod support;

pub use all::*;
pub use any::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[all::ENTRIES, any::ENTRIES];
