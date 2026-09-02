pub mod elementwise;
mod reduction;
mod rounding;
mod trigonometry;

pub use elementwise::*;
pub use reduction::*;
pub use rounding::*;
pub use trigonometry::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = elementwise::ENTRY_GROUPS;
pub(super) const TRIGONOMETRY_ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] =
    &[trigonometry::ENTRIES];
pub(super) const ROUNDING_ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[rounding::ENTRIES];
pub(super) const REDUCTION_ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] =
    reduction::ENTRY_GROUPS;
