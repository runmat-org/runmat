mod bitwise;
mod discrete;
pub mod elementwise;
mod integer_division;
mod linalg;
mod reduction;
mod rounding;
mod trigonometry;

pub use bitwise::*;
pub use discrete::*;
pub use elementwise::*;
pub use integer_division::*;
pub use linalg::*;
pub use reduction::*;
pub use rounding::*;
pub use trigonometry::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    elementwise::extend_entries(entries);
    super::extend_groups(entries, &[bitwise::ENTRIES]);
    super::extend_groups(entries, &[discrete::ENTRIES]);
    super::extend_groups(entries, &[integer_division::ENTRIES]);
    linalg::extend_entries(entries);
    reduction::extend_entries(entries);
    super::extend_groups(entries, &[rounding::ENTRIES]);
    trigonometry::extend_entries(entries);
}
