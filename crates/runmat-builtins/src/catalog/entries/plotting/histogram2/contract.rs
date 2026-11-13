mod parameters_output_handle;

mod errors;
pub use errors::{HISTOGRAM2_ERROR_INTERNAL, HISTOGRAM2_ERROR_INVALID_ARGUMENT};
mod integers;
pub use integers::HISTOGRAM2_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::HISTOGRAM2_DESCRIPTOR;
