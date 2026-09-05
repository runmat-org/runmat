mod cd;
mod path;
mod pwd;

pub use cd::*;
pub use path::*;
pub use pwd::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] =
    &[cd::ENTRIES, path::ENTRIES, pwd::ENTRIES];
