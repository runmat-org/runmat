use crate::indexing::plan::build_index_plan;
use crate::indexing::read_slice::{
    read_complex_slice_from_plan, read_gpu_slice_from_plan, read_sparse_slice_from_plan,
    read_tensor_slice_from_plan,
};
use crate::indexing::selectors::SliceSelector;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

/// Reads scalar paren indices without converting selector values through `f64`.
///
/// The runtime's generic indexing API predates typed integer storage and accepts
/// floating-point selectors. Keep using it for its broader legacy surface, but
/// route storage-aware values through the VM slice plan so both the selector and
/// the selected integer payload retain their exact representation.
pub async fn generic_index(base: &Value, indices: &[usize]) -> Result<Value, RuntimeError> {
    if matches!(indices.len(), 1 | 2) {
        let selectors: Vec<SliceSelector> =
            indices.iter().copied().map(SliceSelector::Scalar).collect();
        match base {
            Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
                let plan = build_index_plan(&selectors, indices.len(), &tensor.shape)?;
                return read_tensor_slice_from_plan(tensor, &plan);
            }
            Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => {
                let plan = build_index_plan(&selectors, indices.len(), &tensor.shape)?;
                return read_complex_slice_from_plan(tensor, &plan);
            }
            Value::SparseTensor(sparse) if sparse.integer_storage().is_some() => {
                let plan = build_index_plan(&selectors, indices.len(), &sparse.shape())?;
                return read_sparse_slice_from_plan(sparse, &plan);
            }
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some() =>
            {
                let plan = build_index_plan(&selectors, indices.len(), &handle.shape)?;
                return read_gpu_slice_from_plan(handle, &plan);
            }
            _ => {}
        }
    }

    let floating_indices: Vec<f64> = indices.iter().map(|&index| index as f64).collect();
    runmat_runtime::perform_indexing(base, &floating_indices).await
}

#[cfg(test)]
mod tests {
    use super::generic_index;
    use futures::executor::block_on;
    use runmat_builtins::{
        ComplexTensor, IntegerComplexStorage, IntegerStorage, SparseTensor, Tensor, Value,
    };

    #[test]
    fn typed_linear_reads_ignore_cleared_f64_mirrors_for_all_integer_classes() {
        macro_rules! assert_read {
            ($storage:ident, $value:expr) => {{
                let tensor =
                    Tensor::new_integer(IntegerStorage::$storage(vec![0, $value]), vec![1, 2])
                        .expect("typed tensor");
                assert_eq!(
                    block_on(generic_index(&Value::Tensor(tensor), &[2])).expect("typed read"),
                    Value::Int(
                        IntegerStorage::$storage(vec![$value])
                            .value_at(0)
                            .expect("value")
                    ),
                );
            }};
        }

        assert_read!(I8, i8::MIN);
        assert_read!(I16, i16::MAX);
        assert_read!(I32, i32::MIN);
        assert_read!(I64, i64::MAX);
        assert_read!(U8, u8::MAX);
        assert_read!(U16, u16::MAX);
        assert_read!(U32, u32::MAX);
        assert_read!(U64, u64::MAX);
    }

    #[test]
    fn typed_complex_and_sparse_linear_reads_preserve_wide_storage() {
        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I64(vec![0, i64::MIN]),
                IntegerStorage::I64(vec![0, i64::MAX]),
            )
            .expect("components"),
            vec![1, 2],
        )
        .expect("complex tensor");
        let Value::ComplexTensor(result) =
            block_on(generic_index(&Value::ComplexTensor(complex), &[2])).expect("complex read")
        else {
            panic!("typed complex read must remain typed");
        };
        assert_eq!(
            result.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![i64::MIN]),
                    IntegerStorage::I64(vec![i64::MAX]),
                )
                .expect("expected components"),
            )
        );

        let sparse = SparseTensor::new_integer(
            1,
            2,
            vec![0, 0, 1],
            vec![0],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .expect("sparse tensor");
        let Value::SparseTensor(result) =
            block_on(generic_index(&Value::SparseTensor(sparse), &[2])).expect("sparse read")
        else {
            panic!("typed sparse read must remain typed");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX]))
        );
    }
}
