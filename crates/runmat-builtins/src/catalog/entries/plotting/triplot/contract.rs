mod parameters_output_handle;

mod errors;
pub use errors::{TRIPLOT_ERROR_INTERNAL, TRIPLOT_ERROR_INVALID_ARGUMENT};
mod extensions;
pub use extensions::{
    TRIPLOT_EXTENSIONS, TRIPLOT_GPU_INPUT_EXTENSION, TRIPLOT_INTEGER_CONNECTIVITY_EXTENSION,
    TRIPLOT_INTEGER_COORDINATE_EXTENSION,
};
mod integers;
pub use integers::TRIPLOT_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::TRIPLOT_DESCRIPTOR;
