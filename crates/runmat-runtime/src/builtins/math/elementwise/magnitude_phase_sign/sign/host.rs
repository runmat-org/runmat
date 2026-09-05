use super::*;

pub(super) fn sign_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("sign", value)
        .map_err(|e| OPERATION.error(&SIGN_ERROR_INVALID_INPUT, e))?;
    Ok(tensor::tensor_into_value(sign_tensor(tensor)?))
}

pub(super) fn sign_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| OPERATION.error(&SIGN_ERROR_INTERNAL, e))?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(sign_real_scalar).collect())
        }
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| sign_real_scalar(f64::from(value)) as f32)
                .collect(),
        ),
        integer => NumericStorage::from_integer_storage(sign_integer_storage(
            &integer
                .into_integer_storage()
                .expect("integer NumericStorage variant"),
        )),
    };
    Tensor::from_numeric_storage(output, shape)
        .map_err(|e| OPERATION.error(&SIGN_ERROR_INTERNAL, e))
}

pub(super) fn sign_integer_scalar(value: IntValue) -> IntValue {
    match value {
        IntValue::I8(value) => IntValue::I8(value.signum()),
        IntValue::I16(value) => IntValue::I16(value.signum()),
        IntValue::I32(value) => IntValue::I32(value.signum()),
        IntValue::I64(value) => IntValue::I64(value.signum()),
        IntValue::U8(value) => IntValue::U8(u8::from(value != 0)),
        IntValue::U16(value) => IntValue::U16(u16::from(value != 0)),
        IntValue::U32(value) => IntValue::U32(u32::from(value != 0)),
        IntValue::U64(value) => IntValue::U64(u64::from(value != 0)),
    }
}

pub(super) fn sign_integer_storage(storage: &IntegerStorage) -> IntegerStorage {
    match storage {
        IntegerStorage::I8(values) => {
            IntegerStorage::I8(values.iter().map(|value| value.signum()).collect())
        }
        IntegerStorage::I16(values) => {
            IntegerStorage::I16(values.iter().map(|value| value.signum()).collect())
        }
        IntegerStorage::I32(values) => {
            IntegerStorage::I32(values.iter().map(|value| value.signum()).collect())
        }
        IntegerStorage::I64(values) => {
            IntegerStorage::I64(values.iter().map(|value| value.signum()).collect())
        }
        IntegerStorage::U8(values) => {
            IntegerStorage::U8(values.iter().map(|value| u8::from(*value != 0)).collect())
        }
        IntegerStorage::U16(values) => {
            IntegerStorage::U16(values.iter().map(|value| u16::from(*value != 0)).collect())
        }
        IntegerStorage::U32(values) => {
            IntegerStorage::U32(values.iter().map(|value| u32::from(*value != 0)).collect())
        }
        IntegerStorage::U64(values) => {
            IntegerStorage::U64(values.iter().map(|value| u64::from(*value != 0)).collect())
        }
    }
}

pub(super) fn sign_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| sign_real_scalar(ch as u32 as f64))
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| OPERATION.error(&SIGN_ERROR_INTERNAL, e))?;
    Ok(Value::Tensor(tensor))
}

pub(super) fn sign_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(re, im)| sign_complex(re, im))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(re, im)| sign_complex_f32(re, im))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(OPERATION.error(
                &SIGN_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| OPERATION.error(&SIGN_ERROR_INTERNAL, e))?;
    Ok(Value::ComplexTensor(tensor))
}

#[inline]
pub(super) fn sign_real_scalar(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

pub(super) fn sign_complex(re: f64, im: f64) -> (f64, f64) {
    if re == 0.0 && im == 0.0 {
        return (0.0, 0.0);
    }
    if re.is_nan() || im.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let re_inf = re.is_infinite();
    let im_inf = im.is_infinite();
    if re_inf || im_inf {
        let real = if re_inf { re.signum() } else { 0.0 };
        let imag = if im_inf { im.signum() } else { 0.0 };
        let norm = (real * real + imag * imag).sqrt();
        if norm == 0.0 {
            return (real, imag);
        }
        return (real / norm, imag / norm);
    }
    let scale = re.abs().max(im.abs());
    if scale == 0.0 {
        return (0.0, 0.0);
    }
    let nr = re / scale;
    let ni = im / scale;
    let magnitude = (nr * nr + ni * ni).sqrt();
    if magnitude == 0.0 {
        (0.0, 0.0)
    } else {
        (nr / magnitude, ni / magnitude)
    }
}

pub(super) fn sign_complex_f32(re: f32, im: f32) -> (f32, f32) {
    if re == 0.0 && im == 0.0 {
        return (0.0, 0.0);
    }
    if re.is_nan() || im.is_nan() {
        return (f32::NAN, f32::NAN);
    }
    let real = if re.is_infinite() { re.signum() } else { 0.0 };
    let imag = if im.is_infinite() { im.signum() } else { 0.0 };
    if re.is_infinite() || im.is_infinite() {
        let norm = (real * real + imag * imag).sqrt();
        return if norm == 0.0 {
            (real, imag)
        } else {
            (real / norm, imag / norm)
        };
    }
    let scale = re.abs().max(im.abs());
    if scale == 0.0 {
        return (0.0, 0.0);
    }
    let real = re / scale;
    let imag = im / scale;
    let magnitude = (real * real + imag * imag).sqrt();
    if magnitude == 0.0 {
        (0.0, 0.0)
    } else {
        (real / magnitude, imag / magnitude)
    }
}
