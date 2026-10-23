mod parameters_inputs_none;
mod controls;
pub use controls::{
    SPHERE_DEFAULT_N, SPHERE_MAX_COORDINATE_ELEMENTS, SPHERE_WHOLE_COUNT_TOLERANCE,
};

mod errors;
pub use errors::{
    SPHERE_ERROR_ARG_COUNT, SPHERE_ERROR_INTERNAL, SPHERE_ERROR_INVALID_N,
    SPHERE_ERROR_OUTPUT_COUNT, SPHERE_ERROR_TOO_LARGE,
};
mod extensions;
pub use extensions::{SPHERE_EXTENSIONS, SPHERE_TYPED_N_EXTENSION};
mod integers;
pub use integers::SPHERE_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::SPHERE_DESCRIPTOR;
