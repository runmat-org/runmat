mod parameters_output;

mod errors;
pub use errors::STACKEDPLOT_ERROR_INVALID_ARGUMENT;
mod extensions;
pub use extensions::{STACKEDPLOT_EXPLICIT_GPU_EXTENSION, STACKEDPLOT_EXTENSIONS};
mod integers;
pub use integers::STACKEDPLOT_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::STACKEDPLOT_DESCRIPTOR;
