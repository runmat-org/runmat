mod documentation;
mod function_dispatch;

pub use function_dispatch::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[function_dispatch::ENTRIES];
