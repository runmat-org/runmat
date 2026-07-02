//! Interpolation builtins and shared piecewise-polynomial helpers.

pub mod interp1;
pub mod interp1q;
pub mod interp2;
pub mod pchip;
pub mod ppval;
pub mod spline;

pub(crate) mod gridded_interpolant;
pub(crate) mod pp;
