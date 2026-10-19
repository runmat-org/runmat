use super::*;

pub(super) fn points_tensor(points: &[[f64; 2]]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(points.len() * 2);
    for col in 0..2 {
        for point in points {
            data.push(point[col]);
        }
    }
    Tensor::new_2d(data, points.len(), 2)
        .map(Value::Tensor)
        .map_err(invalid)
}

pub(super) fn triangles_tensor(triangles: &[[usize; 3]]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(triangles.len() * 3);
    for col in 0..3 {
        for tri in triangles {
            data.push(tri[col] as f64 + 1.0);
        }
    }
    Tensor::new_2d(data, triangles.len(), 3)
        .map(Value::Tensor)
        .map_err(invalid)
}

pub(super) fn constraints_tensor(constraints: &[[usize; 2]]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(constraints.len() * 2);
    for col in 0..2 {
        for constraint in constraints {
            data.push(constraint[col] as f64 + 1.0);
        }
    }
    Tensor::new_2d(data, constraints.len(), 2)
        .map(Value::Tensor)
        .map_err(invalid)
}

pub(super) fn edges_tensor(edges: &[[usize; 2]]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(edges.len() * 2);
    for col in 0..2 {
        for edge in edges {
            data.push(edge[col] as f64 + 1.0);
        }
    }
    Tensor::new_2d(data, edges.len(), 2)
        .map(Value::Tensor)
        .map_err(invalid)
}

pub(super) fn positive_index(value: f64, max: usize) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(invalid("indices must be positive finite integers"));
    }
    let idx = value as usize;
    if idx == 0 || idx > max {
        return Err(invalid("index is out of range"));
    }
    Ok(idx)
}

pub(super) fn tensor_positive_index(
    tensor: &Tensor,
    index: usize,
    max: usize,
) -> BuiltinResult<usize> {
    let value = tensor
        .numeric_value_at(index)
        .ok_or_else(|| invalid("index is out of range"))?;
    if let Some(integer) = value.into_int_value() {
        let idx = integer
            .try_to_usize()
            .ok_or_else(|| invalid("index is out of range"))?;
        if idx == 0 {
            return Err(invalid("indices must be positive finite integers"));
        }
        if idx > max {
            return Err(invalid("index is out of range"));
        }
        return Ok(idx);
    }
    match value {
        NumericScalar::F64(value) => positive_index(value, max),
        NumericScalar::F32(value) => positive_index(f64::from(value), max),
        _ => unreachable!("integer scalar returned by into_int_value"),
    }
}
