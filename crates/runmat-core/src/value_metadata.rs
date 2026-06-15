use runmat_builtins::{LogicalArray, SparseTensor, Value};

/// MATLAB-style class name for a runtime value.
pub fn matlab_class_name(value: &Value) -> String {
    match value {
        Value::Num(_) | Value::ComplexTensor(_) | Value::Complex(_, _) => "double".to_string(),
        Value::Tensor(tensor) => tensor.dtype.class_name().to_string(),
        Value::SparseTensor(_) => "double".to_string(),
        Value::Int(iv) => iv.class_name().to_string(),
        Value::Bool(_) | Value::LogicalArray(_) => "logical".to_string(),
        Value::String(_) | Value::StringArray(_) => "string".to_string(),
        Value::CharArray(_) => "char".to_string(),
        Value::Symbolic(_) => "sym".to_string(),
        Value::Cell(_) => "cell".to_string(),
        Value::Struct(_) => "struct".to_string(),
        Value::GpuTensor(_) => "gpuArray".to_string(),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => "function_handle".to_string(),
        Value::HandleObject(handle) => {
            if handle.class_name.is_empty() {
                "handle".to_string()
            } else {
                handle.class_name.clone()
            }
        }
        Value::Listener(_) => "event.listener".to_string(),
        // Internal destructuring helper; shouldn't surface in user-facing values,
        // but handle it defensively for completeness.
        Value::OutputList(_) => "OutputList".to_string(),
        Value::Object(obj) => obj.class_name.clone(),
        Value::ClassRef(_) => "meta.class".to_string(),
        Value::MException(_) => "MException".to_string(),
    }
}

/// Returns the MATLAB-style shape for the provided value when applicable.
pub fn value_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Complex(_, _)
        | Value::Symbolic(_) => Some(vec![1, 1]),
        Value::LogicalArray(arr) => Some(arr.shape.clone()),
        Value::StringArray(sa) => Some(sa.shape.clone()),
        Value::String(s) => Some(vec![1, s.chars().count()]),
        Value::CharArray(ca) => Some(vec![ca.rows, ca.cols]),
        Value::Tensor(t) => Some(t.shape.clone()),
        Value::SparseTensor(s) => Some(vec![s.rows, s.cols]),
        Value::ComplexTensor(t) => Some(t.shape.clone()),
        Value::Cell(ca) => Some(ca.shape.clone()),
        Value::GpuTensor(handle) => Some(handle.shape.clone()),
        Value::Object(obj) if obj.is_class("datetime") => match obj.properties.get("__serial") {
            Some(Value::Tensor(tensor)) => Some(tensor.shape.clone()),
            Some(Value::Num(_)) => Some(vec![1, 1]),
            _ => None,
        },
        _ => None,
    }
}

/// Returns a MATLAB dtype label for numeric values when available.
pub fn numeric_dtype_label(value: &Value) -> Option<&'static str> {
    match value {
        Value::Num(_) | Value::Complex(_, _) => Some("double"),
        Value::Tensor(t) => Some(t.dtype.class_name()),
        Value::LogicalArray(_) => Some("logical"),
        Value::Int(iv) => Some(iv.class_name()),
        _ => None,
    }
}

/// Rough estimate of the in-memory footprint for the provided value, in bytes.
pub fn approximate_size_bytes(value: &Value) -> Option<u64> {
    Some(match value {
        Value::Num(_) | Value::Int(_) | Value::Complex(_, _) => 8,
        Value::Bool(_) => 1,
        Value::LogicalArray(arr) => arr.data.len() as u64,
        Value::Tensor(t) => (t.data.len() * 8) as u64,
        Value::SparseTensor(s) => sparse_tensor_memory_bytes(s),
        Value::ComplexTensor(t) => (t.data.len() * 16) as u64,
        Value::String(s) => s.len() as u64,
        Value::StringArray(sa) => sa.data.iter().map(|s| s.len() as u64).sum(),
        Value::CharArray(ca) => (ca.rows * ca.cols) as u64,
        _ => return None,
    })
}

/// Rough estimate of the sparse tensor storage footprint, in bytes.
pub fn sparse_tensor_memory_bytes(sparse: &SparseTensor) -> u64 {
    sparse_tensor_memory_bytes_from_lengths(
        sparse.values.len(),
        sparse.row_indices.len(),
        sparse.col_ptrs.len(),
    )
}

fn sparse_tensor_memory_bytes_from_lengths(
    values_len: usize,
    row_indices_len: usize,
    col_ptrs_len: usize,
) -> u64 {
    (values_len as u64)
        .saturating_mul(std::mem::size_of::<f64>() as u64)
        .saturating_add(
            (row_indices_len as u64).saturating_mul(std::mem::size_of::<usize>() as u64),
        )
        .saturating_add((col_ptrs_len as u64).saturating_mul(std::mem::size_of::<usize>() as u64))
}

/// Produce a numeric preview (up to `limit` elements) for scalars and dense arrays.
pub fn preview_numeric_values(value: &Value, limit: usize) -> Option<(Vec<f64>, bool)> {
    match value {
        Value::Num(n) => Some((vec![*n], false)),
        Value::Int(iv) => Some((vec![iv.to_f64()], false)),
        Value::Bool(flag) => Some((vec![if *flag { 1.0 } else { 0.0 }], false)),
        Value::Tensor(t) => Some(preview_f64_slice(&t.data, limit)),
        Value::SparseTensor(s) => Some(preview_sparse_tensor(s, limit)),
        Value::LogicalArray(arr) => Some(preview_logical_slice(arr, limit)),
        Value::StringArray(_) | Value::String(_) | Value::CharArray(_) => None,
        Value::ComplexTensor(_) | Value::Complex(_, _) => None,
        Value::Cell(_)
        | Value::Symbolic(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::OutputList(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::GpuTensor(_) => None,
    }
}

fn preview_f64_slice(data: &[f64], limit: usize) -> (Vec<f64>, bool) {
    if data.len() > limit {
        (data[..limit].to_vec(), true)
    } else {
        (data.to_vec(), false)
    }
}

fn preview_sparse_tensor(sparse: &SparseTensor, limit: usize) -> (Vec<f64>, bool) {
    let total_len = sparse.rows.saturating_mul(sparse.cols);
    let preview_len = total_len.min(limit);
    let mut preview = Vec::with_capacity(preview_len);
    if sparse.rows == 0 {
        return (preview, false);
    }
    for linear_index in 0..preview_len {
        let row = linear_index % sparse.rows;
        let col = linear_index / sparse.rows;
        preview.push(sparse.get(row, col).unwrap_or(0.0));
    }
    (preview, total_len > limit)
}

fn preview_logical_slice(arr: &LogicalArray, limit: usize) -> (Vec<f64>, bool) {
    let truncated = arr.data.len() > limit;
    let mut preview = Vec::with_capacity(arr.data.len().min(limit));
    for value in arr.data.iter().take(limit) {
        preview.push(if *value == 0 { 0.0 } else { 1.0 });
    }
    (preview, truncated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{NumericDType, ObjectInstance, Tensor};

    #[test]
    fn approximate_size_bytes_uses_f64_width_for_integer_dtypes() {
        // Tensor.data is always Vec<f64> (8 bytes/element) regardless of dtype.
        let u8_tensor = Tensor::new_with_dtype(vec![1.0, 2.0, 3.0], vec![3, 1], NumericDType::U8)
            .expect("tensor");
        let u16_tensor = Tensor::new_with_dtype(vec![1.0, 2.0, 3.0], vec![3, 1], NumericDType::U16)
            .expect("tensor");
        let f32_tensor = Tensor::new_with_dtype(vec![1.0, 2.0, 3.0], vec![3, 1], NumericDType::F32)
            .expect("tensor");

        assert_eq!(approximate_size_bytes(&Value::Tensor(u8_tensor)), Some(24));
        assert_eq!(approximate_size_bytes(&Value::Tensor(u16_tensor)), Some(24));
        assert_eq!(approximate_size_bytes(&Value::Tensor(f32_tensor)), Some(24));
    }

    #[test]
    fn sparse_tensor_memory_bytes_uses_saturating_arithmetic() {
        let sparse =
            SparseTensor::new(3, 2, vec![0, 1, 2], vec![0, 2], vec![4.0, 5.0]).expect("sparse");
        let expected = (2 * std::mem::size_of::<f64>())
            + (2 * std::mem::size_of::<usize>())
            + (3 * std::mem::size_of::<usize>());

        assert_eq!(sparse_tensor_memory_bytes(&sparse), expected as u64);
        assert_eq!(
            sparse_tensor_memory_bytes_from_lengths(usize::MAX, usize::MAX, usize::MAX),
            u64::MAX
        );
    }

    #[test]
    fn datetime_object_shape_comes_from_internal_serial_tensor() {
        let mut object = ObjectInstance::new("datetime".to_string());
        object.properties.insert(
            "__serial".to_string(),
            Value::Tensor(Tensor::new(vec![739351.0, 739352.0], vec![2, 1]).expect("tensor")),
        );

        assert_eq!(value_shape(&Value::Object(object)), Some(vec![2, 1]));
    }

    #[test]
    fn sparse_preview_uses_logical_column_major_values() {
        let sparse = SparseTensor::new(3, 3, vec![0, 1, 1, 3], vec![1, 0, 2], vec![4.0, 5.0, 6.0])
            .expect("sparse");

        assert_eq!(
            preview_numeric_values(&Value::SparseTensor(sparse), 9),
            Some((vec![0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 6.0], false))
        );
    }

    #[test]
    fn sparse_preview_truncates_by_logical_element_count() {
        let sparse = SparseTensor::zeros(1000, 1000);

        assert_eq!(
            preview_numeric_values(&Value::SparseTensor(sparse), 3),
            Some((vec![0.0, 0.0, 0.0], true))
        );
    }
}
