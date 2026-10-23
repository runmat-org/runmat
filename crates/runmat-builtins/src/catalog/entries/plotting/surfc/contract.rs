mod parameters_output_handle;

mod errors;
pub use errors::{SURFC_ERROR_INTERNAL, SURFC_ERROR_INVALID_ARGUMENT};
mod integers;
pub use integers::SURFC_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::SURFC_DESCRIPTOR;
