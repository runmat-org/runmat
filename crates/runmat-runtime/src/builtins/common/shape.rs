use runmat_builtins::{Tensor, Value};

use crate::dispatcher::gather_if_needed_async;
use crate::RuntimeError;

/// Return true if a shape should be treated as a scalar.
pub fn is_scalar_shape(shape: &[usize]) -> bool {
    shape.is_empty()
        || (shape.len() == 1 && shape[0] == 1)
        || (shape.len() == 2 && shape[0] == 1 && shape[1] == 1)
}

/// Return the canonical scalar shape.
pub fn canonical_scalar_shape() -> Vec<usize> {
    vec![1, 1]
}

/// Normalize scalar-like shapes to the canonical scalar shape.
pub fn normalize_scalar_shape(shape: &[usize]) -> Vec<usize> {
    if is_scalar_shape(shape) {
        canonical_scalar_shape()
    } else {
        shape.to_vec()
    }
}

/// Normalize a raw shape vector into MATLAB-compatible dimension metadata.
fn normalize_shape(shape: &[usize]) -> Vec<usize> {
    if shape.len() == 1 && shape[0] != 1 {
        return vec![1, shape[0]];
    }
    if is_scalar_shape(shape) {
        return canonical_scalar_shape();
    }
    shape.to_vec()
}

/// Return the MATLAB-visible dimension vector for a runtime value.
#[async_recursion::async_recursion(?Send)]
pub async fn value_dimensions(value: &Value) -> Result<Vec<usize>, RuntimeError> {
    let dims = match value {
        Value::Tensor(t) => normalize_shape(&t.shape),
        Value::ComplexTensor(t) => normalize_shape(&t.shape),
        Value::LogicalArray(la) => normalize_shape(&la.shape),
        Value::StringArray(sa) => normalize_shape(&sa.shape),
        Value::CharArray(ca) => vec![ca.rows, ca.cols],
        Value::Cell(ca) => normalize_shape(&ca.shape),
        Value::GpuTensor(handle) => {
            if handle.shape.is_empty() {
                let gathered = gather_if_needed_async(&Value::GpuTensor(handle.clone())).await?;
                return value_dimensions(&gathered).await;
            }
            normalize_shape(&handle.shape)
        }
        _ => vec![1, 1],
    };
    Ok(dims)
}

/// Compute the total number of elements contained in a runtime value.
#[async_recursion::async_recursion(?Send)]
pub async fn value_numel(value: &Value) -> Result<usize, RuntimeError> {
    let numel = match value {
        Value::Tensor(t) => t.data.len(),
        Value::ComplexTensor(t) => t.data.len(),
        Value::LogicalArray(la) => la.data.len(),
        Value::StringArray(sa) => sa.data.len(),
        Value::CharArray(ca) => ca.rows * ca.cols,
        Value::Cell(ca) => ca.data.len(),
        Value::GpuTensor(handle) => {
            if handle.shape.is_empty() {
                let gathered = gather_if_needed_async(&Value::GpuTensor(handle.clone())).await?;
                return value_numel(&gathered).await;
            }
            handle
                .shape
                .iter()
                .copied()
                .fold(1usize, |acc, dim| acc.saturating_mul(dim))
        }
        _ => 1,
    };
    Ok(numel)
}

/// Compute the dimensionality (NDIMS) of a runtime value, with MATLAB semantics.
pub async fn value_ndims(value: &Value) -> Result<usize, RuntimeError> {
    let dims = value_dimensions(value).await?;
    if dims.len() < 2 {
        Ok(2)
    } else {
        Ok(dims.len())
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
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dims_scalar_defaults_to_one_by_one() {
        assert_eq!(
            block_on(value_dimensions(&Value::Num(5.0))).unwrap(),
            vec![1, 1]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dims_tensor_preserves_rank() {
        let tensor = Tensor::new(vec![0.0; 12], vec![2, 3, 2]).unwrap();
        assert_eq!(
            block_on(value_dimensions(&Value::Tensor(tensor))).unwrap(),
            vec![2, 3, 2]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_gpu_uses_shape_product() {
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![4, 5, 6],
            device_id: 0,
            buffer_id: 1,
        };
        assert_eq!(
            block_on(value_numel(&Value::GpuTensor(handle))).unwrap(),
            120
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dims_to_row_tensor_converts() {
        let tensor = dims_to_row_tensor(&[2, 4, 6]).unwrap();
        assert_eq!(tensor.shape, vec![1, 3]);
        assert_eq!(tensor.data, vec![2.0, 4.0, 6.0]);
    }
}
