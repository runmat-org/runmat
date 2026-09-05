mod pwd;

pub use pwd::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = pwd::ENTRIES;
