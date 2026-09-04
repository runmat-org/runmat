use crate::BuiltinResult;
use runmat_value::{CellArray, Value};

use super::{data, error, sparse, typed_output};

pub(super) fn dense_or_sparse_f64(
    data: Vec<f64>,
    shape: Vec<usize>,
    sparse: bool,
) -> BuiltinResult<Value> {
    sparse::numeric(data, shape, sparse)
}

pub(super) fn from_scalars(
    values: Vec<Value>,
    shape: Vec<usize>,
    sparse: bool,
) -> BuiltinResult<Value> {
    if let Some(integers) = typed_output::exact_integers(&values) {
        return typed_output::integers(integers, shape, sparse);
    }
    if values
        .iter()
        .any(|value| typed_output::exact_integer(value).is_some())
    {
        return Err(typed_output::class_mismatch());
    }
    if values.iter().all(|value| matches!(value, Value::Bool(_))) {
        return typed_output::logical(values, shape, sparse);
    }
    if values
        .iter()
        .all(|value| matches!(value, Value::CharArray(chars) if chars.data.len() == 1))
    {
        return typed_output::characters(values, shape, sparse);
    }
    if values
        .iter()
        .any(|value| matches!(value, Value::CharArray(_)))
    {
        return Err(typed_output::class_mismatch());
    }
    if let Some(dtype) = typed_output::common_floating_dtype(&values) {
        return typed_output::floating(values, shape, sparse, dtype);
    }
    if values
        .iter()
        .any(|value| typed_output::floating_dtype(value).is_some())
    {
        return Err(typed_output::class_mismatch());
    }
    if values
        .iter()
        .all(|value| data::as_numeric_scalar(value).is_some())
    {
        let numeric = values
            .iter()
            .map(|value| data::as_numeric_scalar(value).ok_or_else(typed_output::class_mismatch))
            .collect::<BuiltinResult<Vec<_>>>()?;
        return dense_or_sparse_f64(numeric, shape, sparse);
    }
    if sparse {
        return Err(error::invalid(
            "accumarray: sparse output requires double scalar group results",
        ));
    }
    CellArray::new_with_shape(values, shape)
        .map(Value::Cell)
        .map_err(error::invalid)
}
