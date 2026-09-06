pub(in crate::catalog) mod binary_arithmetic;
mod complex_components;
mod complex_construction;
mod error_functions;
mod exponentials;
mod floating_conversions;
mod gamma_functions;
mod heaviside;
mod hypot;
mod logarithms;
mod magnitude_phase_sign;
mod numeric_conversions;
mod numeric_limits;
mod powers_of_two;
mod roots;
mod support;

pub use binary_arithmetic::*;
pub use complex_components::*;
pub use complex_construction::*;
pub use error_functions::*;
pub use exponentials::*;
pub use floating_conversions::*;
pub use gamma_functions::*;
pub use heaviside::*;
pub use hypot::*;
pub use logarithms::*;
pub use magnitude_phase_sign::*;
pub use numeric_conversions::*;
pub use numeric_limits::*;
pub use powers_of_two::*;
pub use roots::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[
    binary_arithmetic::ENTRIES,
    complex_components::ENTRIES,
    complex_construction::ENTRIES,
    error_functions::ENTRIES,
    exponentials::ENTRIES,
    floating_conversions::ENTRIES,
    gamma_functions::ENTRIES,
    heaviside::ENTRIES,
    hypot::ENTRIES,
    logarithms::ENTRIES,
    magnitude_phase_sign::ENTRIES,
    numeric_conversions::ENTRIES,
    numeric_limits::ENTRIES,
    powers_of_two::ENTRIES,
    roots::ENTRIES,
];
