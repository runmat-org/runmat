use super::*;

pub(super) fn real_tensor_value(
    data: Vec<f64>,
    shape: Vec<usize>,
    label: &str,
) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(Value::Tensor)
        .map_err(|err| internal(format!("lscov: failed to construct {label}: {err}")))
}

pub(super) fn complex_matrix_value(
    matrix: &DMatrix<Complex64>,
    rows: usize,
    cols: usize,
    label: &str,
) -> BuiltinResult<Value> {
    let data = matrix
        .as_slice()
        .iter()
        .map(|value| (value.re, value.im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(data, vec![rows, cols])
        .map_err(|err| internal(format!("lscov: failed to construct {label}: {err}")))?;
    if tensor
        .materialize_f64()
        .iter()
        .all(|(_, im)| im.abs() <= EPS || im.is_nan())
    {
        let real = tensor
            .materialize_f64()
            .iter()
            .map(|(re, _)| *re)
            .collect::<Vec<_>>();
        return real_tensor_value(real, vec![rows, cols], label);
    }
    Ok(complex_tensor_into_value(tensor))
}
