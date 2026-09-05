mod cd;
mod pwd;

pub use cd::*;
pub use pwd::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[cd::ENTRIES, pwd::ENTRIES];
