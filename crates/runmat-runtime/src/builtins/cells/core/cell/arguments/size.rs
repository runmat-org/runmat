use runmat_value::{IntValue, NumericScalar, Tensor, Value};

use super::super::error;

pub(super) async fn parse(
    values: &[Value],
    prototype: Option<&Value>,
) -> crate::BuiltinResult<Vec<usize>> {
    if values.is_empty() {
        return prototype.map_or_else(
            || Ok(vec![0, 0]),
            |value| {
                crate::builtins::common::random_args::shape_from_value(value, "cell")
                    .map_err(error::invalid_input)
            },
        );
    }
    if values.len() == 1 {
        let host = crate::gather_if_needed_async(&values[0]).await?;
        return single(&host);
    }
    let mut dimensions = Vec::with_capacity(values.len());
    for value in values {
        let host = crate::gather_if_needed_async(value).await?;
        dimensions.push(scalar(&host)?);
    }
    Ok(dimensions)
}

fn single(value: &Value) -> crate::BuiltinResult<Vec<usize>> {
    match value {
        Value::Int(_) | Value::Num(_) => scalar(value).map(|size| vec![size, size]),
        Value::Tensor(tensor) => vector(tensor),
        _ => Err(error::invalid_input(
            "size must be a numeric scalar or row vector",
        )),
    }
}

fn vector(tensor: &Tensor) -> crate::BuiltinResult<Vec<usize>> {
    if tensor.is_empty() {
        return Ok(vec![0, 0]);
    }
    if !row_vector(&tensor.shape) {
        return Err(error::invalid_size("size vector must be a row vector"));
    }
    let mut dimensions = (0..tensor.len())
        .map(|index| {
            tensor
                .numeric_value_at(index)
                .ok_or_else(|| error::internal("size storage is inconsistent"))
                .and_then(numeric)
        })
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    if dimensions.len() == 1 {
        dimensions.push(dimensions[0]);
    }
    Ok(dimensions)
}

fn scalar(value: &Value) -> crate::BuiltinResult<usize> {
    match value {
        Value::Int(value) => integer(value),
        Value::Num(value) => floating(*value),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor
            .numeric_value_at(0)
            .ok_or_else(|| error::internal("scalar size storage is inconsistent"))
            .and_then(numeric),
        Value::Tensor(_) => Err(error::invalid_size("size inputs must be scalar")),
        _ => Err(error::invalid_input("size inputs must be numeric scalars")),
    }
}

fn numeric(value: NumericScalar) -> crate::BuiltinResult<usize> {
    match value {
        NumericScalar::F64(value) => floating(value),
        NumericScalar::F32(value) => floating(f64::from(value)),
        value => value
            .into_int_value()
            .ok_or_else(|| error::internal("numeric size is neither floating nor integer"))
            .and_then(|value| integer(&value)),
    }
}

fn integer(value: &IntValue) -> crate::BuiltinResult<usize> {
    usize::try_from(value.to_i128().max(0))
        .map_err(|_| error::invalid_size("requested size exceeds platform limits"))
}

fn floating(value: f64) -> crate::BuiltinResult<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(error::invalid_size("size inputs must be finite integers"));
    }
    let value = value.max(0.0);
    if value > (1u64 << 53) as f64 || value >= usize::MAX as f64 {
        return Err(error::invalid_size(
            "requested size exceeds platform limits",
        ));
    }
    Ok(value as usize)
}

fn row_vector(shape: &[usize]) -> bool {
    matches!(shape, [] | [_] | [0 | 1, _])
}
