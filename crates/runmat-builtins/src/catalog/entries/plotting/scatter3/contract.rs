mod parameters_output_handle;
mod parameters_inputs_ax_x_y_z;
mod parameters_inputs_ax_x_y_z_props;
mod signatures;
mod errors;
mod descriptor;
mod integers;

pub use descriptor::SCATTER3_DESCRIPTOR;
pub use errors::{SCATTER3_ERROR_INTERNAL, SCATTER3_ERROR_INVALID_ARGUMENT};
pub use integers::SCATTER3_INTEGER_CAPABILITIES;
