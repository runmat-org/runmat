use super::*;
pub(super) fn tensor_matrix_storage(value: Value) -> BuiltinResult<(NumericStorage, usize, usize)> {
    match value {
        Value::Tensor(t) => {
            if t.shape().len() > 2 {
                return Err(pagefun_error_with_detail(
                    &PAGEFUN_ERROR_RESULT_TYPE,
                    "operator returned an array with more than two dimensions",
                ));
            }
            let canonical = canonical_matrix_shape(t.shape());
            let rows = canonical[0];
            let cols = canonical[1];
            let data = t
                .into_numeric_storage()
                .map_err(|error| pagefun_error_with_detail(&PAGEFUN_ERROR_RESULT_TYPE, error))?;
            if rows * cols != data.len() {
                return Err(pagefun_error_with_detail(
                    &PAGEFUN_ERROR_RESULT_SHAPE,
                    "result size mismatch",
                ));
            }
            Ok((data, rows, cols))
        }
        value @ (Value::Num(_) | Value::Int(_)) => {
            let tensor = tensor_utils::value_into_tensor_for(BUILTIN_NAME, value)
                .map_err(|error| pagefun_error_with_detail(&PAGEFUN_ERROR_RESULT_TYPE, &error))?;
            let data = tensor
                .into_numeric_storage()
                .map_err(|error| pagefun_error_with_detail(&PAGEFUN_ERROR_RESULT_TYPE, error))?;
            Ok((data, 1, 1))
        }
        other => Err(pagefun_error_with_detail(
            &PAGEFUN_ERROR_RESULT_TYPE,
            format!(
                "expected numeric matrix result, received {}",
                other.type_name()
            ),
        )),
    }
}

pub(super) fn empty_storage_like(storage: &NumericStorage, capacity: usize) -> NumericStorage {
    macro_rules! empty {
        ($variant:ident) => {
            NumericStorage::$variant(Vec::with_capacity(capacity))
        };
    }
    match storage {
        NumericStorage::F64(_) => empty!(F64),
        NumericStorage::F32(_) => empty!(F32),
        NumericStorage::I8(_) => empty!(I8),
        NumericStorage::I16(_) => empty!(I16),
        NumericStorage::I32(_) => empty!(I32),
        NumericStorage::I64(_) => empty!(I64),
        NumericStorage::U8(_) => empty!(U8),
        NumericStorage::U16(_) => empty!(U16),
        NumericStorage::U32(_) => empty!(U32),
        NumericStorage::U64(_) => empty!(U64),
    }
}

pub(super) fn extend_same_class(
    target: &mut NumericStorage,
    source: NumericStorage,
) -> BuiltinResult<()> {
    macro_rules! extend {
        ($target:ident, $source:ident) => {{
            $target.extend($source);
            Ok(())
        }};
    }
    match (target, source) {
        (NumericStorage::F64(target), NumericStorage::F64(source)) => extend!(target, source),
        (NumericStorage::F32(target), NumericStorage::F32(source)) => extend!(target, source),
        (NumericStorage::I8(target), NumericStorage::I8(source)) => extend!(target, source),
        (NumericStorage::I16(target), NumericStorage::I16(source)) => extend!(target, source),
        (NumericStorage::I32(target), NumericStorage::I32(source)) => extend!(target, source),
        (NumericStorage::I64(target), NumericStorage::I64(source)) => extend!(target, source),
        (NumericStorage::U8(target), NumericStorage::U8(source)) => extend!(target, source),
        (NumericStorage::U16(target), NumericStorage::U16(source)) => extend!(target, source),
        (NumericStorage::U32(target), NumericStorage::U32(source)) => extend!(target, source),
        (NumericStorage::U64(target), NumericStorage::U64(source)) => extend!(target, source),
        (target, source) => Err(pagefun_error_with_detail(
            &PAGEFUN_ERROR_RESULT_TYPE,
            format!(
                "operator returned inconsistent page classes ({} then {})",
                target.class_name(),
                source.class_name()
            ),
        )),
    }
}

pub(super) fn complex_matrix_data(value: Value) -> BuiltinResult<ComplexMatrixData> {
    match value {
        Value::ComplexTensor(t) => {
            if t.shape().len() > 2 {
                return Err(pagefun_error_with_detail(
                    &PAGEFUN_ERROR_RESULT_TYPE,
                    "operator returned an array with more than two dimensions",
                ));
            }
            let canonical = canonical_matrix_shape(t.shape());
            let rows = canonical[0];
            let cols = canonical[1];
            if rows * cols != t.materialize_f64().len() {
                return Err(pagefun_error_with_detail(
                    &PAGEFUN_ERROR_RESULT_SHAPE,
                    "result size mismatch",
                ));
            }
            Ok((t.materialize_f64(), rows, cols, t.numeric_dtype()))
        }
        Value::Complex(re, im) => Ok((vec![(re, im)], 1, 1, NumericDType::F64)),
        other => Err(pagefun_error_with_detail(
            &PAGEFUN_ERROR_RESULT_TYPE,
            format!(
                "expected complex matrix result, received {}",
                other.type_name()
            ),
        )),
    }
}
