use runmat_value::{Tensor, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::RuntimeError;

mod object;
pub(crate) use object::array_payload_storage_matches;
#[cfg(test)]
#[path = "shape/object_tests.rs"]
mod object_tests;

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

/// Return MATLAB's effective rank after ignoring trailing singleton dimensions.
/// MATLAB arrays retain a minimum visible rank of two.
pub fn effective_rank(shape: &[usize]) -> usize {
    shape
        .iter()
        .rposition(|extent| *extent != 1)
        .map_or(2, |index| (index + 1).max(2))
}

/// Compare array dimensions, ignoring stored singleton axes beyond the
/// two-dimensional language minimum. Internal rank-zero/rank-one carriers
/// retain their exact shape rather than becoming public matrix arrays.
pub fn array_shapes_equal(left: &[usize], right: &[usize]) -> bool {
    if left.len() < 2 || right.len() < 2 {
        return left == right;
    }
    let left_rank = effective_rank(left);
    let right_rank = effective_rank(right);
    left_rank == right_rank && left[..left_rank] == right[..right_rank]
}

/// Borrowed geometry or the existing two-axis scalar/sparse representation.
/// No payload, rank vector or recursive future allocation is required.
#[derive(Clone, Copy, Debug)]
pub enum ValueDimensions<'a> {
    Stored(&'a [usize]),
    Inline([usize; 2]),
}
impl ValueDimensions<'_> {
    pub fn as_slice(&self) -> &[usize] {
        match self {
            Self::Stored(shape) => shape,
            Self::Inline(shape) => shape,
        }
    }
    fn normalized(self) -> Self {
        let shape = self.as_slice();
        if shape.len() == 1 && shape[0] != 1 {
            return Self::Inline([1, shape[0]]);
        }
        if is_scalar_shape(shape) {
            return Self::Inline([1, 1]);
        }
        self
    }
}
impl std::ops::Deref for ValueDimensions<'_> {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        self.as_slice()
    }
}
impl AsRef<[usize]> for ValueDimensions<'_> {
    fn as_ref(&self) -> &[usize] {
        self.as_slice()
    }
}
impl PartialEq for ValueDimensions<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl Eq for ValueDimensions<'_> {}

/// Exact stored geometry used by ordinary indexed receivers. This preserves
/// rank-zero/rank-one storage policy; visible dimensions normalize separately.
pub fn stored_dimensions_borrowed(value: &Value) -> ValueDimensions<'_> {
    match value {
        Value::Tensor(t) => ValueDimensions::Stored(t.shape()),
        Value::SparseTensor(t) => ValueDimensions::Inline([t.rows, t.cols]),
        Value::ComplexTensor(t) => ValueDimensions::Stored(t.shape()),
        Value::LogicalArray(t) => ValueDimensions::Stored(&t.shape),
        Value::StringArray(t) => ValueDimensions::Stored(&t.shape),
        Value::SymbolicArray(t) => ValueDimensions::Stored(&t.shape),
        Value::CharArray(t) => ValueDimensions::Stored(t.shape()),
        Value::Cell(t) => ValueDimensions::Stored(&t.shape),
        Value::StructArray(t) => ValueDimensions::Stored(t.shape()),
        Value::ObjectArray(t) => ValueDimensions::Stored(t.shape()),
        Value::GpuTensor(t) => ValueDimensions::Stored(t.shape()),
        _ => ValueDimensions::Inline([1, 1]),
    }
}

/// Canonical MATLAB-visible geometry. Declared object payloads are validated
/// numeric/logical storage, never recursive Object values.
pub fn value_dimensions_borrowed(value: &Value) -> Result<ValueDimensions<'_>, RuntimeError> {
    match value {
        Value::Object(object) => {
            if let Some(shape) = object.optimization_model_shape() {
                return Ok(ValueDimensions::Stored(shape));
            }
            if let Some(shape) = object.optimization_variable_shape() {
                return Ok(ValueDimensions::Stored(shape));
            }
            match runmat_builtins::standard_class_array_shape(object.class_name()) {
                Some(declaration) => object::dimensions(object, declaration),
                None => Ok(ValueDimensions::Inline([1, 1])),
            }
        }
        // Character geometry was historically reported exactly as stored.
        Value::CharArray(_) => Ok(stored_dimensions_borrowed(value)),
        _ => Ok(stored_dimensions_borrowed(value).normalized()),
    }
}

/// Ordinary owned-vector compatibility adapter. Managed consumers retain the
/// borrowed view directly in their existing admitted owner instead.
pub async fn value_dimensions(value: &Value) -> Result<Vec<usize>, RuntimeError> {
    Ok(value_dimensions_borrowed(value)?.as_slice().to_vec())
}

/// Compute the total number of elements contained in a runtime value.
pub async fn value_numel(value: &Value) -> Result<usize, RuntimeError> {
    let numel = match value {
        Value::Tensor(t) => tensor_utils::tensor_element_len(t),
        Value::SparseTensor(t) => t.rows.saturating_mul(t.cols),
        Value::ComplexTensor(t) => tensor_utils::complex_tensor_element_len(t),
        Value::LogicalArray(la) => la.data.len(),
        Value::StringArray(sa) => sa.data().len(),
        Value::SymbolicArray(sa) => sa.data().len(),
        Value::CharArray(ca) => ca.data().len(),
        Value::Cell(ca) => ca.data().len(),
        Value::StructArray(array) => array.len(),
        Value::ObjectArray(array) => array.len(),
        Value::GpuTensor(handle) => handle
            .shape()
            .iter()
            .copied()
            .fold(1usize, |acc, dim| acc.saturating_mul(dim)),
        Value::Object(_) => value_dimensions_borrowed(value)?
            .iter()
            .copied()
            .try_fold(1usize, |count, dimension| count.checked_mul(dimension))
            .ok_or_else(|| {
                crate::build_runtime_error("array element count exceeds the structural range")
                    .build()
            })?,
        _ => 1,
    };
    Ok(numel)
}

/// Compute the dimensionality (NDIMS) of a runtime value, with MATLAB semantics.
pub async fn value_ndims(value: &Value) -> Result<usize, RuntimeError> {
    let dims = value_dimensions_borrowed(value)?;
    Ok(effective_rank(&dims))
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
    use runmat_value::IntegerStorage;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dims_scalar_defaults_to_one_by_one() {
        assert_eq!(
            block_on(value_dimensions(&Value::Num(5.0))).unwrap(),
            vec![1, 1]
        );
    }

    #[test]
    fn borrowed_dimensions_preserve_storage_and_visible_rank_policies() {
        let value = Value::Tensor(Tensor::new(vec![0.0; 6], vec![2, 3]).unwrap());
        let Value::Tensor(tensor) = &value else {
            unreachable!()
        };
        let stored = stored_dimensions_borrowed(&value);
        let visible = value_dimensions_borrowed(&value).unwrap();
        assert_eq!(stored.as_slice().as_ptr(), tensor.shape().as_ptr());
        assert_eq!(visible.as_slice().as_ptr(), tensor.shape().as_ptr());
        let value = Value::Tensor(Tensor::new(vec![0.0; 3], vec![3]).unwrap());
        assert_eq!(stored_dimensions_borrowed(&value).as_slice(), &[3]);
        assert_eq!(
            value_dimensions_borrowed(&value).unwrap().as_slice(),
            &[1, 3]
        );
        assert_eq!(
            ValueDimensions::Stored(&[2, 3]),
            ValueDimensions::Inline([2, 3])
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
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn dimensions_normalize_low_rank_real_and_complex_arrays() {
        for value in [
            Value::Tensor(Tensor::new(vec![1.0], Vec::new()).expect("rank-zero real")),
            Value::ComplexTensor(
                runmat_value::ComplexTensor::new(vec![(1.0, 2.0)], Vec::new())
                    .expect("rank-zero complex"),
            ),
        ] {
            assert_eq!(block_on(value_dimensions(&value)).unwrap(), vec![1, 1]);
        }
        for value in [
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2]).expect("rank-one real")),
            Value::ComplexTensor(
                runmat_value::ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2])
                    .expect("rank-one complex"),
            ),
        ] {
            assert_eq!(block_on(value_dimensions(&value)).unwrap(), vec![1, 2]);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_gpu_uses_shape_product() {
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let allocation = owner
            .allocate_zeroed(
                vec![4, 5, 6],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        assert_eq!(
            block_on(value_numel(&Value::GpuTensor(allocation.handle))).unwrap(),
            120
        );
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn empty_gpu_shape_is_scalar_metadata_without_provider_access() {
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let allocation = owner
            .allocate_zeroed(
                vec![],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let value = Value::GpuTensor(allocation.handle);
        assert_eq!(block_on(value_dimensions(&value)).unwrap(), vec![1, 1]);
        assert_eq!(block_on(value_numel(&value)).unwrap(), 1);
        assert_eq!(allocation.probe.retirements(), 0);
        drop(value);
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_reads_typed_integer_storage_length_exactly() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 7, 9]), vec![1, 3]).unwrap();

        assert_eq!(block_on(value_numel(&Value::Tensor(tensor))).unwrap(), 3);
    }

    #[test]
    fn numel_reads_typed_complex_integer_storage_length_exactly() {
        let storage = runmat_value::IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX, 7]),
            IntegerStorage::U64(vec![0, 0]),
        )
        .unwrap();
        let tensor = runmat_value::ComplexTensor::new_integer(storage, vec![1, 2]).unwrap();

        assert_eq!(
            block_on(value_numel(&Value::ComplexTensor(tensor))).unwrap(),
            2
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dims_to_row_tensor_converts() {
        let tensor = dims_to_row_tensor(&[2, 4, 6]).unwrap();
        assert_eq!(tensor.shape(), &[1, 3]);
        assert_eq!(tensor.materialize_f64(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn effective_rank_ignores_only_trailing_singletons() {
        assert_eq!(effective_rank(&[2, 3]), 2);
        assert_eq!(effective_rank(&[2, 3, 1, 1]), 2);
        assert_eq!(effective_rank(&[1, 1, 4, 1]), 3);
        assert_eq!(effective_rank(&[1, 1, 1]), 2);
        assert!(array_shapes_equal(&[2, 1, 1], &[2, 1]));
        assert!(array_shapes_equal(&[0, 3, 1], &[0, 3]));
        assert!(!array_shapes_equal(&[2, 1, 2], &[2, 1]));
        assert!(!array_shapes_equal(&[1, 2], &[2, 1]));
        assert!(!array_shapes_equal(&[], &[1, 1]));
        assert!(!array_shapes_equal(&[1], &[1, 1]));
    }
}
