mod parameters_output_handle;

mod errors;
pub use errors::{POLARPLOT_ERROR_INTERNAL, POLARPLOT_ERROR_INVALID_ARGUMENT};
mod extensions;
pub use extensions::{POLARPLOT_COMPLEX_INTEGER_EXTENSION, POLARPLOT_EXTENSIONS};
mod integers;
pub use integers::POLARPLOT_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::POLARPLOT_DESCRIPTOR;
