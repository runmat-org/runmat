mod bitwise;
pub mod elementwise;
mod reduction;
mod rounding;
mod trigonometry;

pub use bitwise::*;
pub use elementwise::*;
pub use reduction::*;
pub use rounding::*;
pub use trigonometry::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(entries, elementwise::ENTRY_GROUPS);
    super::extend_groups(entries, &[bitwise::ENTRIES]);
    super::extend_groups(entries, reduction::ENTRY_GROUPS);
    super::extend_groups(entries, &[rounding::ENTRIES]);
    super::extend_groups(entries, &[trigonometry::ENTRIES]);
}
