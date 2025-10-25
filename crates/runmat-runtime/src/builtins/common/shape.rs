use runmat_builtins::{Tensor, Value};

/// Normalize a raw shape vector into MATLAB-compatible dimension metadata.
fn normalize_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => shape.to_vec(),
    }
}

/// Return the MATLAB-visible dimension vector for a runtime value.
pub fn value_dimensions(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(t) => normalize_shape(&t.shape),
        Value::ComplexTensor(t) => normalize_shape(&t.shape),
        Value::LogicalArray(la) => normalize_shape(&la.shape),
        Value::StringArray(sa) => normalize_shape(&sa.shape),
        Value::CharArray(ca) => vec![ca.rows, ca.cols],
        Value::Cell(ca) => vec![ca.rows, ca.cols],
        Value::GpuTensor(handle) => {
            if handle.shape.is_empty() {
                if let Some(provider) = runmat_accelerate_api::provider() {
                    if let Ok(host) = provider.download(handle) {
                        return normalize_shape(&host.shape);
                    }
                }
                vec![1, 1]
            } else {
                normalize_shape(&handle.shape)
            }
        }
        _ => vec![1, 1],
    }
}

/// Compute the total number of elements contained in a runtime value.
pub fn value_numel(value: &Value) -> usize {
    match value {
        Value::Tensor(t) => t.data.len(),
        Value::ComplexTensor(t) => t.data.len(),
        Value::LogicalArray(la) => la.data.len(),
        Value::StringArray(sa) => sa.data.len(),
        Value::CharArray(ca) => ca.rows * ca.cols,
        Value::Cell(ca) => ca.data.len(),
        Value::GpuTensor(handle) => {
            if handle.shape.is_empty() {
                if let Some(provider) = runmat_accelerate_api::provider() {
                    if let Ok(host) = provider.download(handle) {
                        return host.data.len();
                    }
                }
                1
            } else {
                handle
                    .shape
                    .iter()
                    .copied()
                    .fold(1usize, |acc, dim| acc.saturating_mul(dim))
            }
        }
        _ => 1,
    }
}

/// Compute the dimensionality (NDIMS) of a runtime value, with MATLAB semantics.
pub fn value_ndims(value: &Value) -> usize {
    let dims = value_dimensions(value);
    if dims.len() < 2 {
        2
    } else {
        dims.len()
    }
}

/// Convert a dimension vector into a 1×N tensor encoded as `f64`.
pub fn dims_to_row_tensor(dims: &[usize]) -> Result<Tensor, String> {
    let len = dims.len();
    let data: Vec<f64> = dims.iter().map(|&d| d as f64).collect();
    let shape = if len == 0 { vec![1, 0] } else { vec![1, len] };
    Tensor::new(data, shape).map_err(|e| format!("shape::dims_to_row_tensor: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_scalar_defaults_to_one_by_one() {
        assert_eq!(value_dimensions(&Value::Num(5.0)), vec![1, 1]);
    }

    #[test]
    fn dims_tensor_preserves_rank() {
        let tensor = Tensor::new(vec![0.0; 12], vec![2, 3, 2]).unwrap();
        assert_eq!(value_dimensions(&Value::Tensor(tensor)), vec![2, 3, 2]);
    }

    #[test]
    fn numel_gpu_uses_shape_product() {
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![4, 5, 6],
            device_id: 0,
            buffer_id: 1,
        };
        assert_eq!(value_numel(&Value::GpuTensor(handle)), 120);
    }

    #[test]
    fn dims_to_row_tensor_converts() {
        let tensor = dims_to_row_tensor(&[2, 4, 6]).unwrap();
        assert_eq!(tensor.shape, vec![1, 3]);
        assert_eq!(tensor.data, vec![2.0, 4.0, 6.0]);
    }
}
