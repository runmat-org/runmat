//! Catalog identities of the `fea.SignalKnot` typed policy class.

use crate::*;

mod continuous;
mod jump;
#[cfg(test)]
mod tests;

pub(super) const KNOT_ROLE: &str = concat!(
    "A signal knot is one authored point of a tabulated time history. Its abscissa is a ",
    "time in seconds and its ordinate is in the canonical SI unit of the signal that will",
    " own it; the knot itself carries no unit or quantity until it is passed to a ",
    "`fromTimeKnots` constructor. Numeric inputs accept double and every built-in integer",
    " class and are stored as binary64; a negative-zero abscissa is stored as 0.",
);
pub(super) const KNOT_STUDY: &str = concat!(
    "Collect knots, in increasing time order, in a row or column cell array and pass it ",
    "to a time-history constructor such as `fea.ForceSignal.fromTimeKnots(id, knots, ",
    "interpolation, extrapolation)`. Every knot of one signal must have the same ordinate",
    " shape: scalar for scalar quantities such as pressure or temperature, three ",
    "components for vector quantities such as force or displacement.",
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &continuous::FEA_SIGNAL_KNOT_CONTINUOUS_CATALOG_ENTRY,
    &jump::FEA_SIGNAL_KNOT_JUMP_CATALOG_ENTRY,
];
