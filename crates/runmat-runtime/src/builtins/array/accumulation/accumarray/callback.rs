use std::collections::BTreeMap;

use runmat_value::{IntValue, IntegerStorage, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::table::select_rows;
use crate::{call_feval_async_with_outputs, BuiltinResult};

use super::{data, error, output};

pub(super) async fn evaluate(
    data: Value,
    groups: BTreeMap<usize, Vec<usize>>,
    shape: Vec<usize>,
    output_len: usize,
    function: Value,
    fill: Option<Value>,
    sparse: bool,
) -> BuiltinResult<Value> {
    if sparse && !data::is_double(&data) {
        return Err(error::invalid(
            "accumarray: sparse output requires double input data",
        ));
    }
    let mut computed = BTreeMap::new();
    for (linear, rows) in groups {
        let group = select_rows(&data, &rows)?;
        computed.insert(linear, invoke(function.clone(), group).await?);
    }
    let prototype = computed.values().next();
    let fill = match fill {
        Some(value) => scalar(value)?,
        None => prototype
            .map(zero_like)
            .transpose()?
            .unwrap_or(Value::Num(0.0)),
    };
    if sparse && data::as_numeric_scalar(&fill) != Some(0.0) {
        return Err(error::invalid(
            "accumarray: sparse output requires a zero fill value",
        ));
    }
    let mut values = vec![fill; output_len];
    for (linear, value) in computed {
        values[linear] = value;
    }
    output::from_scalars(values, shape, sparse)
}

async fn invoke(function: Value, values: Value) -> BuiltinResult<Value> {
    let result = call_feval_async_with_outputs(function, &[values], 1)
        .await
        .map_err(|source| error::callback("accumarray: callback failed", source))?;
    scalar(match result {
        Value::OutputList(mut values) if values.len() == 1 => values.remove(0),
        other => other,
    })
}

fn scalar(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(&tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .value_at(0)
                    .map(Value::Int)
                    .ok_or_else(|| error::invalid("accumarray: empty scalar result"));
            }
            Ok(Value::Tensor(tensor))
        }
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(Value::Bool(array.data[0] != 0)),
        Value::Cell(cell) if cell.data.len() == 1 => cell
            .data
            .first()
            .cloned()
            .ok_or_else(|| error::invalid("accumarray: empty scalar cell result")),
        Value::CharArray(chars) if chars.data.len() == 1 => Ok(Value::CharArray(chars)),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(value),
        other => Err(error::invalid(format!(
            "accumarray: group function must return a scalar, got {other:?}"
        ))),
    }
}

fn zero_like(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::Int(value) => integer_zero(value),
        Value::Num(_) => Ok(Value::Num(0.0)),
        Value::Bool(_) => Ok(Value::Bool(false)),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            runmat_value::Tensor::new_with_dtype(vec![0.0], vec![1, 1], tensor.numeric_dtype())
                .map(Value::Tensor)
                .map_err(error::invalid)
        }
        _ => Err(error::invalid(
            "accumarray: explicit fill value required for nonnumeric group results",
        )),
    }
}

fn integer_zero(value: &IntValue) -> BuiltinResult<Value> {
    IntegerStorage::from_scalar(value.clone())
        .zeros_like(1)
        .value_at(0)
        .map(Value::Int)
        .ok_or_else(|| error::invalid("accumarray: could not construct integer fill"))
}
