mod parameters_output_handle;

mod errors;
pub use errors::{IMAGESC_ERROR_INTERNAL, IMAGESC_ERROR_INVALID_ARGUMENT};
mod extensions;
pub use extensions::{IMAGESC_EXTENSIONS, IMAGESC_FOUR_CHANNEL_EXTENSION};
mod integers;
pub use integers::IMAGESC_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::IMAGESC_DESCRIPTOR;
