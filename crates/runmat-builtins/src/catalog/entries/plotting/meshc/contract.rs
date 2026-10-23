mod parameters_output_handle;

mod errors;
pub use errors::{MESHC_ERROR_INTERNAL, MESHC_ERROR_INVALID_ARGUMENT};
mod integers;
pub use integers::MESHC_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::MESHC_DESCRIPTOR;
