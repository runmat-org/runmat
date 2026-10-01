mod ppval;
mod pchip;
mod interp1q;
mod interp2;
mod interp1;
mod gridded_interpolant;
mod spline;
mod piecewise_inference;

pub use gridded_interpolant::*;
pub use interp1::*;
pub use interp1q::*;
pub use interp2::*;
pub use pchip::*;
pub use ppval::*;
pub use spline::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(ppval::ENTRIES.iter().copied());
    values.extend(pchip::ENTRIES.iter().copied());
    values.extend(interp1q::ENTRIES.iter().copied());
    values.extend(interp2::ENTRIES.iter().copied());
    values.extend(interp1::ENTRIES.iter().copied());
    gridded_interpolant::extend_entries(values);
    values.extend(spline::ENTRIES.iter().copied());
}
