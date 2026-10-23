mod parameters_output_counts;

mod errors;
pub use errors::{HIST_ERROR_INTERNAL, HIST_ERROR_INVALID_ARGUMENT, HIST_ERROR_TOO_MANY_OUTPUTS};
mod extensions;
pub use extensions::{
    HIST_EXTENSIONS, HIST_INTEGER_CENTERS_EXTENSION, HIST_INTEGER_DATA_EXTENSION,
    HIST_MODERN_OPTIONS_EXTENSION,
};
mod integers;
pub use integers::HIST_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::HIST_DESCRIPTOR;
