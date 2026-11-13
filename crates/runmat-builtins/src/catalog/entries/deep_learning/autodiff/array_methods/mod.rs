//! Differentiable host dlarray method forms.

mod contract;
mod inference;
mod shared;
mod minus;
mod mtimes;
mod plus;
mod rdivide;
mod sum;
mod times;
#[cfg(test)]
mod tests;

pub use contract::{
    DLARRAY_MINUS_DESCRIPTOR, DLARRAY_MTIMES_DESCRIPTOR, DLARRAY_PLUS_DESCRIPTOR,
    DLARRAY_RDIVIDE_DESCRIPTOR, DLARRAY_SUM_DESCRIPTOR, DLARRAY_TIMES_DESCRIPTOR,
    INVALID_INPUT as DLARRAY_METHOD_INVALID_INPUT,
};
pub use minus::DLARRAY_MINUS_CATALOG_ENTRY;
pub use mtimes::DLARRAY_MTIMES_CATALOG_ENTRY;
pub use plus::DLARRAY_PLUS_CATALOG_ENTRY;
pub use rdivide::DLARRAY_RDIVIDE_CATALOG_ENTRY;
pub use sum::DLARRAY_SUM_CATALOG_ENTRY;
pub use times::DLARRAY_TIMES_CATALOG_ENTRY;
