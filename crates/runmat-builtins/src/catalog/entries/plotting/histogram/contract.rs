mod parameters_output_handle;

mod parameters_inputs_ax_x_bins_namevalue;

mod errors;
pub use errors::{HISTOGRAM_ERROR_INTERNAL, HISTOGRAM_ERROR_INVALID_ARGUMENT};
mod integers;
pub use integers::HISTOGRAM_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::HISTOGRAM_DESCRIPTOR;
