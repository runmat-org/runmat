mod creation;
mod documentation;
mod introspection;

pub use creation::*;
pub use introspection::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[creation::ENTRIES];

pub(super) const INTROSPECTION_ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] =
    introspection::ENTRY_GROUPS;
