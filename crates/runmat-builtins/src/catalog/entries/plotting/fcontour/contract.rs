mod parameters_output_handle;

mod errors;
pub use errors::{
    FCONTOUR_ERROR_EVALUATION, FCONTOUR_ERROR_INTERNAL, FCONTOUR_ERROR_INVALID_ARGUMENT,
};
mod extensions;
pub use extensions::{
    FCONTOUR_EXTENSIONS, FCONTOUR_INTEGER_LINE_COLOR_EXTENSION,
    FCONTOUR_POSITIONAL_LEVEL_EXTENSION, FCONTOUR_RESIDENT_NUMERIC_EXTENSION,
};
mod integers;
pub use integers::FCONTOUR_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::FCONTOUR_DESCRIPTOR;
