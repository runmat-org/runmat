mod parameters_output_handle;

mod errors;
pub use errors::{POLARHISTOGRAM_ERROR_INTERNAL, POLARHISTOGRAM_ERROR_INVALID_ARGUMENT};
mod integers;
pub use integers::POLARHISTOGRAM_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::POLARHISTOGRAM_DESCRIPTOR;
