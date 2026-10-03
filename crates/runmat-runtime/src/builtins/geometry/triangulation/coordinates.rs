use super::*;

pub(super) fn matrix_points(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<[f64; 2]>> {
    let tensor = numeric_tensor(value, builtin)?;
    if !has_only_trailing_singletons(&tensor.shape) {
        return Err(invalid(format!(
            "{builtin}: points must be an N-by-2 matrix"
        )));
    }
    if tensor.cols == 3 {
        return Err(unsupported(
            "3-D DelaunayTri triangulation is not yet supported",
        ));
    }
    if tensor.cols != 2 {
        return Err(invalid(format!(
            "{builtin}: points must be an N-by-2 matrix"
        )));
    }
    let values = tensor_utils::tensor_values_f64_cow(&tensor);
    let mut points = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let x = values[row];
        let y = values[row + tensor.rows];
        if !x.is_finite() || !y.is_finite() {
            return Err(invalid(format!("{builtin}: points must be finite")));
        }
        points.push([x, y]);
    }
    Ok(points)
}

pub(super) fn vector_points(
    x: &Value,
    y: &Value,
    builtin: &'static str,
) -> BuiltinResult<Vec<[f64; 2]>> {
    let x = numeric_tensor(x, builtin)?;
    let y = numeric_tensor(y, builtin)?;
    if !is_vector(&x) || !is_vector(&y) || x.len() != y.len() {
        return Err(invalid(format!(
            "{builtin}: x and y must be vectors with the same length"
        )));
    }
    let x_values = tensor_utils::tensor_values_f64_cow(&x);
    let y_values = tensor_utils::tensor_values_f64_cow(&y);
    x_values
        .iter()
        .zip(y_values.iter())
        .map(|(x, y)| {
            if !x.is_finite() || !y.is_finite() {
                Err(invalid(format!("{builtin}: points must be finite")))
            } else {
                Ok([*x, *y])
            }
        })
        .collect()
}

pub(super) fn constraints_from_value(value: &Value) -> BuiltinResult<Vec<[usize; 2]>> {
    let tensor = numeric_tensor(value, BUILTIN_NAME)?;
    if tensor.is_empty() || tensor.rows == 0 {
        return Ok(Vec::new());
    }
    if tensor.cols != 2 {
        return Err(invalid("DelaunayTri: constraints must be an M-by-2 matrix"));
    }
    let mut constraints = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        constraints.push([
            tensor_positive_index(&tensor, row, usize::MAX)?,
            tensor_positive_index(&tensor, row + tensor.rows, usize::MAX)?,
        ]);
    }
    Ok(constraints)
}

pub(super) fn connectivity_from_value(
    value: &Value,
    point_count: Option<usize>,
) -> BuiltinResult<Vec<[usize; 3]>> {
    let tensor = numeric_tensor(value, DELAUNAY_TRI_CLASS.display_name())?;
    if tensor.cols != 3 {
        return Err(invalid("DelaunayTri connectivity must be an M-by-3 matrix"));
    }
    let mut triangles = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        triangles.push([
            tensor_positive_index(&tensor, row, usize::MAX)? - 1,
            tensor_positive_index(&tensor, row + tensor.rows, usize::MAX)? - 1,
            tensor_positive_index(&tensor, row + 2 * tensor.rows, usize::MAX)? - 1,
        ]);
    }
    if let Some(point_count) = point_count {
        for tri in &triangles {
            if tri.iter().any(|idx| *idx >= point_count) {
                return Err(invalid(
                    "DelaunayTri connectivity references a missing point",
                ));
            }
        }
    }
    Ok(triangles)
}

pub(super) fn numeric_tensor(value: &Value, builtin: &'static str) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.clone()),
        Value::Num(value) => Tensor::new(vec![*value], vec![1, 1]).map_err(invalid),
        Value::Int(value) => {
            Tensor::new_integer(IntegerStorage::from_scalar(value.clone()), vec![1, 1])
                .map_err(invalid)
        }
        other => Err(invalid(format!(
            "{builtin}: expected numeric input, got {other:?}"
        ))),
    }
}

pub(super) fn looks_like_point_matrix(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.cols >= 2)
        || matches!(value, Value::GpuTensor(handle) if handle.shape().get(1).copied().unwrap_or(1) >= 2)
}

pub(super) fn looks_like_constraints(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.cols == 2)
}

pub(super) fn is_vector_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if is_vector(tensor))
        || matches!(value, Value::GpuTensor(handle) if has_only_trailing_singletons(handle.shape()) && (handle.shape().first().copied().unwrap_or(1) == 1 || handle.shape().get(1).copied().unwrap_or(1) == 1))
        || matches!(value, Value::Num(_) | Value::Int(_))
}

pub(super) fn is_vector(tensor: &Tensor) -> bool {
    has_only_trailing_singletons(&tensor.shape) && (tensor.rows == 1 || tensor.cols == 1)
}

pub(super) fn has_only_trailing_singletons(shape: &[usize]) -> bool {
    shape.iter().skip(2).all(|extent| *extent == 1)
}
