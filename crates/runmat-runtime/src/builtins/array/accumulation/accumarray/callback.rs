use std::collections::BTreeMap;

use runmat_value::{IntValue, IntegerStorage, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::table::select_rows;
use crate::BuiltinResult;

use super::{data, error, output};

pub(super) struct Request {
    pub data: Value,
    pub groups: BTreeMap<usize, Vec<usize>>,
    pub shape: Vec<usize>,
    pub output_len: usize,
    pub function: Value,
    pub fill: Option<Value>,
    pub sparse: bool,
}

pub(super) async fn evaluate(
    runtime: Option<&crate::context::RuntimeContext>,
    request: Request,
) -> BuiltinResult<crate::execution::RootedValue> {
    let Request {
        data,
        groups,
        shape,
        output_len,
        function,
        fill,
        sparse,
    } = request;
    let function = crate::execution::RootedValue::in_optional_context(function, runtime)?;
    if sparse && !data::is_double(&data) {
        return Err(error::invalid(
            "accumarray: sparse output requires double input data",
        ));
    }
    let mut computed = BTreeMap::new();
    for (linear, rows) in groups {
        let group = select_rows(&data, &rows)?;
        computed.insert(linear, invoke(runtime, function.clone(), group).await?);
    }
    let prototype = computed.values().next();
    let fill = match fill {
        Some(value) => scalar(value)?,
        None => prototype
            .map(|value| zero_like(value.value()))
            .transpose()?
            .unwrap_or(Value::Num(0.0)),
    };
    if sparse && data::as_numeric_scalar(&fill) != Some(0.0) {
        return Err(error::invalid(
            "accumarray: sparse output requires a zero fill value",
        ));
    }
    let fill = crate::execution::RootedValue::in_optional_context(fill, runtime)?;
    // Numerical/output assembly scratch is distinct from the argument ABI.
    // Keep every computed source and fill rooted until final publication.
    let mut values = Vec::with_capacity(output_len);
    for linear in 0..output_len {
        if let Some(value) = computed.get(&linear) {
            values.push(value.value().clone());
        } else {
            values.push(fill.value().clone());
        }
    }
    let output = output::from_scalars(values, shape, sparse)?;
    crate::execution::RootedValue::in_optional_context(output, runtime)
}

async fn invoke(
    runtime: Option<&crate::context::RuntimeContext>,
    function: crate::execution::RootedValue,
    values: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let arguments =
        crate::call::BuiltinArguments::from_borrowed(std::slice::from_ref(&values), runtime)?;
    let result = crate::call::feval::invoke_rooted_target(runtime, function, arguments, 1, None)
        .await
        .map_err(|source| error::callback("accumarray: callback failed", source))?;
    result.with_owned_sequence(|sequence| {
        let value = scalar(match sequence.into_legacy_value() {
            Value::OutputList(values) if values.len() == 1 => {
                crate::sequence::into_first_output(values)?.expect("one callback output")
            }
            other => other,
        })?;
        crate::execution::RootedValue::in_optional_context(value, runtime)
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
        Value::Cell(cell) if cell.data().len() == 1 => cell
            .data()
            .first()
            .cloned()
            .ok_or_else(|| error::invalid("accumarray: empty scalar cell result")),
        Value::CharArray(chars) if chars.data().len() == 1 => Ok(Value::CharArray(chars)),
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
