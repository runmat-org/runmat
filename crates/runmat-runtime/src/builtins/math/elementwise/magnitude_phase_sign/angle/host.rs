use super::*;

pub(super) fn angle_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("angle", value)
        .map_err(|e| OPERATION.error(&ANGLE_ERROR_INVALID_INPUT, e))?;
    Ok(tensor::tensor_into_value(angle_tensor(tensor)?))
}

pub(super) fn angle_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| OPERATION.error(&ANGLE_ERROR_INTERNAL, error))?;
    let mapped = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(|re| angle_scalar(re, 0.0)).collect())
        }
        NumericStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(|re| 0.0_f32.atan2(re)).collect())
        }
        _ => {
            return Err(OPERATION.error(
                &ANGLE_ERROR_INVALID_INPUT,
                "expected single or double input",
            ))
        }
    };
    Tensor::from_numeric_storage(mapped, shape)
        .map_err(|e| OPERATION.error(&ANGLE_ERROR_INTERNAL, e))
}

pub(super) fn angle_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => NumericStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| angle_scalar(real, imag))
                .collect(),
        ),
        ComplexStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| imag.atan2(real))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(OPERATION.error(
                &ANGLE_ERROR_INVALID_INPUT,
                "expected single or double input",
            ))
        }
    };
    let tensor = Tensor::from_numeric_storage(storage, shape)
        .map_err(|e| OPERATION.error(&ANGLE_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
pub(super) fn angle_scalar(re: f64, im: f64) -> f64 {
    im.atan2(re)
}
