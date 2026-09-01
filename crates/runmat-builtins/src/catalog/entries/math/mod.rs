pub mod elementwise;
mod trigonometry;

pub use elementwise::*;
pub use trigonometry::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = elementwise::ENTRY_GROUPS;
pub(super) const TRIGONOMETRY_ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] =
    &[trigonometry::ENTRIES];
