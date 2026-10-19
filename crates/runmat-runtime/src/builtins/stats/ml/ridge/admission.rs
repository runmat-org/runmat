use super::{invalid_argument, NAME};
use crate::builtins::common::tensor;
use crate::{gather_if_needed_async, BuiltinResult};
use runmat_builtins::RidgeScaling;
use runmat_value::{NumericScalar, Tensor, Value};

pub(super) async fn value_to_tensor(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value).await?;
    tensor::value_into_tensor_for(NAME, gathered)
        .map_err(|err| invalid_argument(format!("ridge: {err}")))
}

fn visible_shape(value: &Value) -> Option<(usize, usize, usize, usize)> {
    let shape: &[usize] = match value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::LogicalArray(array) => &array.shape,
        Value::GpuTensor(handle) => handle.shape(),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => return Some((1, 1, 2, 1)),
        _ => return None,
    };
    let rows = shape.first().copied().unwrap_or(1);
    let columns = shape.get(1).copied().unwrap_or(1);
    let count = shape
        .iter()
        .try_fold(1usize, |total, &dimension| total.checked_mul(dimension))?;
    Some((rows, columns, shape.len(), count))
}

pub(super) fn preflight_data(y: &Value, x: &Value, k: &Value) -> BuiltinResult<()> {
    for value in [y, x, k] {
        if !matches!(
            value,
            Value::Num(_)
                | Value::Int(_)
                | Value::Bool(_)
                | Value::Tensor(_)
                | Value::LogicalArray(_)
                | Value::GpuTensor(_)
        ) {
            return Err(invalid_argument(
                "ridge: expected numeric or logical values",
            ));
        }
    }
    let response = visible_shape(y);
    if let Some((rows, columns, rank, _)) = response {
        if rank > 2 || (rows != 1 && columns != 1) {
            return Err(invalid_argument("ridge: y must be a vector"));
        }
    }
    if let Some((rows, _, rank, _)) = visible_shape(x) {
        if rank > 2 {
            return Err(invalid_argument("ridge: X must be a 2-D numeric matrix"));
        }
        if response.is_some_and(|(_, _, _, count)| count != rows) {
            return Err(invalid_argument(
                "ridge: length(y) must match the number of rows in X",
            ));
        }
    }
    if let Some((rows, columns, rank, count)) = visible_shape(k) {
        if rank > 2 || (rows != 1 && columns != 1) {
            return Err(invalid_argument("ridge: k must be a vector"));
        }
        if count == 0 {
            return Err(invalid_argument(
                "ridge: k must contain at least one ridge parameter",
            ));
        }
    }
    match k {
        Value::Num(value) if !value.is_finite() || *value < 0.0 => Err(invalid_argument(
            "ridge: k values must be nonnegative finite scalars",
        )),
        Value::Int(value) if value.to_i128() < 0 => Err(invalid_argument(
            "ridge: k values must be nonnegative finite scalars",
        )),
        Value::Tensor(tensor) => {
            for index in 0..tensor.host_buffer().len() {
                let Some(scalar) = tensor.host_buffer().value_at(index) else {
                    continue;
                };
                let value = match scalar {
                    NumericScalar::F64(value) => value,
                    NumericScalar::F32(value) => f64::from(value),
                    integer => integer
                        .into_int_value()
                        .map_or(f64::NAN, |value| value.to_f64()),
                };
                if !value.is_finite() || value < 0.0 {
                    return Err(invalid_argument(
                        "ridge: k values must be nonnegative finite scalars",
                    ));
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn decode_host_scaled(value: &Value) -> BuiltinResult<Option<RidgeScaling>> {
    let parsed = match value {
        Value::Bool(value) => Some(RidgeScaling::from_bool(*value)),
        Value::Num(value) => RidgeScaling::from_float(*value),
        Value::Int(value) => RidgeScaling::from_integer(value.to_i128()),
        Value::Tensor(tensor) => {
            if tensor.host_buffer().len() != 1 {
                return Err(invalid_argument(format!(
                    "ridge: scaled must be 0, 1, false, or true, got {:?}",
                    value
                )));
            }
            let value = tensor
                .host_buffer()
                .value_at(0)
                .ok_or_else(|| invalid_argument("ridge: scaled must contain one value"))?;
            match value {
                NumericScalar::F64(value) => RidgeScaling::from_float(value),
                NumericScalar::F32(value) => RidgeScaling::from_float(f64::from(value)),
                integer => integer
                    .into_int_value()
                    .and_then(|value| RidgeScaling::from_integer(value.to_i128())),
            }
        }
        Value::GpuTensor(handle) => {
            if handle
                .shape()
                .iter()
                .try_fold(1usize, |total, &dimension| total.checked_mul(dimension))
                != Some(1)
            {
                return Err(invalid_argument("ridge: scaled must be scalar"));
            }
            return Ok(None);
        }
        other => {
            return Err(invalid_argument(format!(
                "ridge: scaled must be 0, 1, false, or true, got {other:?}"
            )));
        }
    };
    parsed
        .map(Some)
        .ok_or_else(|| invalid_argument("ridge: scaled must be 0, 1, false, or true"))
}

pub(super) fn preflight_scaled(value: &Value) -> BuiltinResult<Option<RidgeScaling>> {
    decode_host_scaled(value)
}

pub(super) async fn parse_resident_scaled(value: &Value) -> BuiltinResult<RidgeScaling> {
    let gathered = gather_if_needed_async(value).await?;
    decode_host_scaled(&gathered)?.ok_or_else(|| invalid_argument("ridge: scaled must be scalar"))
}
