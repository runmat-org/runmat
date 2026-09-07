pub(in crate::catalog) mod binary_arithmetic;
pub(in crate::catalog) mod bsxfun;
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
pub(in crate::catalog) mod rescale;
mod roots;
mod support;
pub(in crate::catalog) mod typecast;

pub use binary_arithmetic::*;
pub use bsxfun::{
    BSXFUN_CATALOG_ENTRY, BSXFUN_DESCRIPTOR, BSXFUN_ERROR_FUNCTION_ERROR, BSXFUN_ERROR_INTERNAL,
    BSXFUN_ERROR_INVALID_FUNCTION, BSXFUN_ERROR_INVALID_INPUT, BSXFUN_ERROR_SIZE_MISMATCH,
    BSXFUN_EXTENSIONS, BSXFUN_INTEGER_CAPABILITIES,
};
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
pub use rescale::{
    RESCALE_CATALOG_ENTRY, RESCALE_DESCRIPTOR, RESCALE_ERROR_INTERNAL,
    RESCALE_ERROR_INVALID_ARGUMENT, RESCALE_ERROR_INVALID_INPUT, RESCALE_ERROR_SIZE_MISMATCH,
    RESCALE_INTEGER_CAPABILITIES,
};
pub use roots::*;
pub use typecast::{
    TYPECAST_CATALOG_ENTRY, TYPECAST_DESCRIPTOR, TYPECAST_ERROR_GPU_UNSUPPORTED,
    TYPECAST_ERROR_INTERNAL, TYPECAST_ERROR_INVALID_ARGUMENT, TYPECAST_ERROR_INVALID_INPUT,
    TYPECAST_INTEGER_CAPABILITIES,
};

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[
    binary_arithmetic::ENTRIES,
    bsxfun::ENTRIES,
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
    rescale::ENTRIES,
    typecast::ENTRIES,
];
