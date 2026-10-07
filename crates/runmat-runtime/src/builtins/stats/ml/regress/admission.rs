use super::*;

pub(super) async fn gather_tensor(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value).await?;
    tensor::value_into_tensor_for(NAME, gathered).map_err(|err| invalid(format!("regress: {err}")))
}

pub(super) enum AlphaHeader {
    Host(Value),
    Resident(Value),
}

pub(super) enum PreparedAlpha {
    Host(f64),
    Resident(Value),
}

impl AlphaHeader {
    pub(super) fn source(&self) -> &Value {
        match self {
            Self::Host(source) | Self::Resident(source) => source,
        }
    }

    pub(super) fn after_integer_boundary(self) -> BuiltinResult<PreparedAlpha> {
        match self {
            Self::Host(source) => parse_alpha_host(&source).map(PreparedAlpha::Host),
            Self::Resident(source) => Ok(PreparedAlpha::Resident(source)),
        }
    }
}

impl PreparedAlpha {
    pub(super) async fn finish(self) -> BuiltinResult<f64> {
        match self {
            Self::Host(value) => Ok(value),
            Self::Resident(source) => {
                let gathered = gather_if_needed_async(&source).await?;
                parse_alpha_host(&gathered)
            }
        }
    }
}

fn visible_shape(value: &Value) -> BuiltinResult<(usize, usize, usize, usize)> {
    let shape: &[usize] = match value {
        Value::Tensor(value) => value.shape(),
        Value::LogicalArray(value) => &value.shape,
        Value::GpuTensor(handle)
            if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real =>
        {
            handle.shape()
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => return Ok((1, 1, 2, 1)),
        _ => return Err(invalid("regress: expected numeric or logical values")),
    };
    let count = shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| invalid("regress: input dimensions overflow"))?;
    let (rows, columns) = regress_matrix_dimensions(shape);
    Ok((rows, columns, shape.len(), count))
}

pub(super) fn regress_matrix_dimensions(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(super) fn preflight_data(y: &Value, x: &Value, outputs: Option<usize>) -> BuiltinResult<()> {
    let (y_rows, y_columns, y_rank, y_count) = visible_shape(y)?;
    let (x_rows, x_columns, x_rank, _) = visible_shape(x)?;
    if outputs == Some(0) {
        return Ok(());
    }
    if y_rank > 2 || (y_rows != 1 && y_columns != 1) {
        return Err(invalid("regress: y must be a vector"));
    }
    if x_rank > 2 {
        return Err(invalid("regress: X must be a 2-D numeric matrix"));
    }
    if x_rows != y_count {
        return Err(invalid(
            "regress: length(y) must match the number of rows in X",
        ));
    }
    if x_columns == 0 {
        return Err(invalid("regress: X must contain at least one column"));
    }
    if y_count == 0 {
        return Err(invalid(
            "regress: at least one complete observation is required",
        ));
    }
    ensure_input_budget(x_rows, x_columns)
}

pub(super) fn prepare_alpha(source: Value) -> BuiltinResult<AlphaHeader> {
    match &source {
        Value::GpuTensor(handle) => {
            let count = handle
                .shape()
                .iter()
                .try_fold(1usize, |count, &extent| count.checked_mul(extent))
                .ok_or_else(|| invalid("regress: alpha dimensions overflow"))?;
            if count != 1
                || handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real
                || handle.view().is_logical()
            {
                return Err(invalid("regress: alpha must be a numeric scalar"));
            }
            Ok(AlphaHeader::Resident(source))
        }
        Value::Num(_) | Value::Int(_) => Ok(AlphaHeader::Host(source)),
        Value::Tensor(tensor) if tensor.len() == 1 => Ok(AlphaHeader::Host(source)),
        _ => Err(invalid(format!(
            "regress: alpha must be a numeric scalar, got {source:?}"
        ))),
    }
}

fn parse_alpha_host(value: &Value) -> BuiltinResult<f64> {
    let alpha = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) => {
            if tensor.len() != 1 {
                return Err(invalid(format!(
                    "regress: alpha must be a numeric scalar, got {:?}",
                    value
                )));
            }
            tensor::tensor_value_f64(tensor, 0)
        }
        other => {
            return Err(invalid(format!(
                "regress: alpha must be a numeric scalar, got {other:?}"
            )));
        }
    };
    if !(0.0..1.0).contains(&alpha) {
        return Err(invalid("regress: alpha must be between 0 and 1"));
    }
    Ok(alpha)
}
