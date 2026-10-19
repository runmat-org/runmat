use runmat_builtins::COMPLEX_ERROR_SIZE_MISMATCH;
use runmat_value::Tensor;

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::super::error;

pub(in crate::builtins::math::elementwise::complex_construction::complex) fn compatible(
    real: &Tensor,
    imaginary: &Tensor,
) -> BuiltinResult<Vec<usize>> {
    compatible_shapes(real.shape(), imaginary.shape())
}

pub(in crate::builtins::math::elementwise::complex_construction::complex) fn compatible_shapes(
    real: &[usize],
    imaginary: &[usize],
) -> BuiltinResult<Vec<usize>> {
    let scalar =
        |shape: &[usize]| shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)) == Some(1);
    if real == imaginary || scalar(imaginary) {
        Ok(real.to_vec())
    } else if scalar(real) {
        Ok(imaginary.to_vec())
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
