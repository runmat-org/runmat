mod ppval;
mod pchip;
mod spline;
mod piecewise_inference;

pub use pchip::*;
pub use ppval::*;
pub use spline::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(ppval::ENTRIES.iter().copied());
    values.extend(pchip::ENTRIES.iter().copied());
    values.extend(spline::ENTRIES.iter().copied());
}
