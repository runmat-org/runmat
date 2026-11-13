mod parameters_output_handle;
mod parameters_inputs_ax_y;
mod signatures;
mod errors;
mod extensions;
mod integers;
mod descriptor;

pub use descriptor::AREA_DESCRIPTOR;
pub use errors::{AREA_ERROR_INTERNAL, AREA_ERROR_INVALID_ARGUMENT};
pub use extensions::{AREA_EXTENSIONS, AREA_LINESPEC_EXTENSION};
pub use integers::AREA_INTEGER_CAPABILITIES;
