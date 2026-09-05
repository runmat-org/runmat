use runmat_builtins::COMPLEX_ERROR_INTERNAL;
use runmat_value::{ComplexStorage, NumericStorage, Tensor, Value};

use crate::BuiltinResult;

use super::super::error;
use super::{complex_value, shape};

pub(super) fn compose(real: &Tensor, imaginary: &Tensor) -> BuiltinResult<Value> {
    let output_shape = shape::compatible(real, imaginary)?;
    let real = real
        .clone()
        .into_numeric_storage()
        .map_err(|detail| error(&COMPLEX_ERROR_INTERNAL, detail))?;
    let imaginary = imaginary
        .clone()
        .into_numeric_storage()
        .map_err(|detail| error(&COMPLEX_ERROR_INTERNAL, detail))?;
    let storage = match (real, imaginary) {
        (NumericStorage::F64(real), NumericStorage::F64(imaginary)) => {
            ComplexStorage::F64(pair(&real, &imaginary)?.into())
        }
        (NumericStorage::F32(real), NumericStorage::F32(imaginary)) => {
            ComplexStorage::F32(pair(&real, &imaginary)?.into())
        }
        (NumericStorage::F64(real), NumericStorage::F32(imaginary)) => {
            let real = real
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            ComplexStorage::F32(pair(&real, &imaginary)?.into())
        }
        (NumericStorage::F32(real), NumericStorage::F64(imaginary)) => {
            let imaginary = imaginary
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            ComplexStorage::F32(pair(&real, &imaginary)?.into())
        }
        _ => {
            return Err(error(
                &COMPLEX_ERROR_INTERNAL,
                "integer input bypassed exact composition",
            ))
        }
    };
    complex_value(storage, output_shape)
}

fn pair<T: Copy>(real: &[T], imaginary: &[T]) -> BuiltinResult<Vec<(T, T)>> {
    match (real, imaginary) {
        ([], []) => Ok(Vec::new()),
        ([real], imaginary) => Ok(imaginary
            .iter()
            .copied()
            .map(|imaginary| (*real, imaginary))
            .collect()),
        (real, [imaginary]) => Ok(real
            .iter()
            .copied()
            .map(|real| (real, *imaginary))
            .collect()),
        (real, imaginary) if real.len() == imaginary.len() => Ok(real
            .iter()
            .copied()
            .zip(imaginary.iter().copied())
            .collect()),
        _ => Err(error(
            &COMPLEX_ERROR_INTERNAL,
            "validated component sizes diverged during composition",
        )),
    }
}
