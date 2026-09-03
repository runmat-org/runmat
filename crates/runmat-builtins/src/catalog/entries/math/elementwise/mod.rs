mod complex_components;
mod error_functions;
mod exponentials;
mod floating_conversions;
mod gamma;
mod gammaln;
mod hypot;
mod logarithms;
mod magnitude_phase_sign;
mod numeric_conversions;
mod numeric_limits;
mod roots;
mod support;

pub use complex_components::*;
pub use error_functions::*;
pub use exponentials::*;
pub use floating_conversions::*;
pub use gamma::*;
pub use gammaln::*;
pub use hypot::*;
pub use logarithms::*;
pub use magnitude_phase_sign::*;
pub use numeric_conversions::*;
pub use numeric_limits::*;
pub use roots::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[
    complex_components::ENTRIES,
    error_functions::ENTRIES,
    exponentials::ENTRIES,
    floating_conversions::ENTRIES,
    gamma::ENTRIES,
    gammaln::ENTRIES,
    hypot::ENTRIES,
    logarithms::ENTRIES,
    magnitude_phase_sign::ENTRIES,
    numeric_conversions::ENTRIES,
    numeric_limits::ENTRIES,
    roots::ENTRIES,
];
