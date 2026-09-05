use runmat_builtins::COMPLEX_ERROR_SIZE_MISMATCH;
use runmat_value::Tensor;

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::super::error;

pub(super) fn compatible(real: &Tensor, imaginary: &Tensor) -> BuiltinResult<Vec<usize>> {
    if real.shape == imaginary.shape || is_scalar(imaginary) {
        Ok(real.shape.clone())
    } else if is_scalar(real) {
        Ok(imaginary.shape.clone())
    } else {
        Err(error(
            &COMPLEX_ERROR_SIZE_MISMATCH,
            "real and imaginary parts must have the same size, unless one input is scalar",
        ))
    }
}

pub(super) fn is_scalar(tensor: &Tensor) -> bool {
    tensor::is_scalar_tensor(tensor)
}
