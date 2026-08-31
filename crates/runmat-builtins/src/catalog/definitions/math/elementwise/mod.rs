mod absolute_value;
mod angle;
mod complex_components;
mod exp;
mod expm1;
mod floating_conversions;
mod log1p;
mod log2;
mod logarithms;
mod numeric_conversions;
mod numeric_limits;
mod sign;
mod support;

pub use absolute_value::*;
pub use angle::*;
pub use complex_components::*;
pub use exp::*;
pub use expm1::*;
pub use floating_conversions::*;
pub use log1p::*;
pub use log2::*;
pub use logarithms::*;
pub use numeric_conversions::*;
pub use numeric_limits::*;
pub use sign::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[
    absolute_value::ENTRIES,
    angle::ENTRIES,
    complex_components::ENTRIES,
    exp::ENTRIES,
    expm1::ENTRIES,
    floating_conversions::ENTRIES,
    log1p::ENTRIES,
    log2::ENTRIES,
    logarithms::ENTRIES,
    numeric_conversions::ENTRIES,
    numeric_limits::ENTRIES,
    sign::ENTRIES,
];
