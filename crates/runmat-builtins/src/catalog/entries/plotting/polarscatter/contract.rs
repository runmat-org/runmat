mod parameters_output_handle;

mod errors;
pub use errors::{POLARSCATTER_ERROR_INTERNAL, POLARSCATTER_ERROR_INVALID_ARGUMENT};
mod integers;
pub use integers::POLARSCATTER_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::POLARSCATTER_DESCRIPTOR;
