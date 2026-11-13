mod parameters_param_s;

mod errors;
pub use errors::{
    PLOTMATRIX_ERROR_TOO_MANY_OUTPUTS, PLOTMATRIX_ERR_INTERNAL, PLOTMATRIX_ERR_INVALID_ARGUMENT,
};
mod integers;
pub use integers::PLOTMATRIX_INTEGER_CAPABILITIES;
mod signatures;

mod descriptor;
pub use descriptor::PLOTMATRIX_DESCRIPTOR;
