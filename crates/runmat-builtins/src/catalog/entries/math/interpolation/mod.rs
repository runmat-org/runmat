mod ppval;
mod pchip;
mod interp1q;
mod interp1;
mod spline;
mod piecewise_inference;

pub use interp1::*;
pub use interp1q::*;
pub use pchip::*;
pub use ppval::*;
pub use spline::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(ppval::ENTRIES.iter().copied());
    values.extend(pchip::ENTRIES.iter().copied());
    values.extend(interp1q::ENTRIES.iter().copied());
    values.extend(interp1::ENTRIES.iter().copied());
    values.extend(spline::ENTRIES.iter().copied());
}
