use crate::indexing::plan::{build_index_plan, IndexPlan};
use crate::indexing::selectors::{build_slice_selectors, SliceSelector};
use crate::RuntimeError;
use runmat_value::{ComplexTensor, SparseTensor, StringArray, Tensor, Value};

pub(super) mod payload;
pub(in crate::indexing) mod sparse;
pub use sparse::SparseReadZeroPolicy;

fn map_slice_shape_error(err: impl std::fmt::Display) -> RuntimeError {
    crate::runtime_error::semantic_error(
        "ShapeMismatch",
        format!("shape mismatch for slice result: {err}"),
    )
}

pub async fn read_tensor_slice_1d(
    tensor: &Tensor,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    read_tensor_slice_nd(tensor, 1, colon_mask, end_mask, numeric).await
}

pub fn try_tensor_slice_2d_fast_path(
    tensor: &Tensor,
    dims: usize,
    selectors: &[SliceSelector],
) -> Result<Option<Value>, RuntimeError> {
    let context = crate::context::legacy::active();
    try_tensor_slice_2d_fast_path_in_context(context.as_ref(), tensor, dims, selectors)
}

pub fn try_tensor_slice_2d_fast_path_in_context(
    context: Option<&crate::context::RuntimeContext>,
    tensor: &Tensor,
    dims: usize,
    selectors: &[SliceSelector],
) -> Result<Option<Value>, RuntimeError> {
    if dims != 2 {
        return Ok(None);
    }
    let rows = tensor.shape().first().copied().unwrap_or(1);
    let cols = tensor.shape().get(1).copied().unwrap_or(1);
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let output_shape = |rows, cols| {
        crate::indexing::plan::admission::shape(context, 2, |dimension| {
            Ok(if dimension == 0 { rows } else { cols })
        })
    };
    match (&selectors[0], &selectors[1]) {
        (SliceSelector::Colon, SliceSelector::Scalar(j)) => {
            let j0 = *j - 1;
            if j0 >= cols {
                return Err(crate::runtime_error::semantic_error(
                    "IndexOutOfBounds",
                    "Index out of bounds",
                ));
            }
            let start = j0 * rows;
            payload::real(
                tensor,
                rows,
                |position| start + position,
                || output_shape(rows, 1),
                context,
            )
            .map(Some)
        }
        (SliceSelector::Scalar(i), SliceSelector::Colon) => {
            let i0 = *i - 1;
            if i0 >= rows {
                return Err(crate::runtime_error::semantic_error(
                    "IndexOutOfBounds",
                    "Index out of bounds",
                ));
            }
            payload::real(
                tensor,
                cols,
                |position| i0 + position * rows,
                || output_shape(1, cols),
                context,
            )
            .map(Some)
        }
        (SliceSelector::Colon, SliceSelector::Indices(js)) => {
            for &j in js {
                owner.check()?;
                let j0 = j - 1;
                if j0 >= cols {
                    return Err(crate::runtime_error::semantic_error(
                        "IndexOutOfBounds",
                        "Index out of bounds",
                    ));
                }
            }
            let count = rows
                .checked_mul(js.len())
                .ok_or_else(|| map_slice_shape_error("selection size overflow"))?;
            payload::real(
                tensor,
                count,
                |position| (js[position / rows] - 1) * rows + position % rows,
                || output_shape(rows, js.len()),
                context,
            )
            .map(Some)
        }
        (SliceSelector::Indices(is), SliceSelector::Colon) => {
            if cols != 0 {
                for &i in is {
                    owner.check()?;
                    let i0 = i - 1;
                    if i0 >= rows {
                        return Err(crate::runtime_error::semantic_error(
                            "IndexOutOfBounds",
                            "Index out of bounds",
                        ));
                    }
                }
            }
            let count = is
                .len()
                .checked_mul(cols)
                .ok_or_else(|| map_slice_shape_error("selection size overflow"))?;
            payload::real(
                tensor,
                count,
                |position| is[position % is.len()] - 1 + (position / is.len()) * rows,
                || output_shape(is.len(), cols),
                context,
            )
            .map(Some)
        }
        _ => Ok(None),
    }
}

pub async fn read_tensor_slice_nd(
    tensor: &Tensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    read_tensor_slice_nd_in_context(
        context.as_ref(),
        tensor,
        dims,
        colon_mask,
        end_mask,
        numeric,
    )
    .await
}

pub async fn read_tensor_slice_nd_in_context(
    context: Option<&crate::context::RuntimeContext>,
    tensor: &Tensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let selectors = match context {
        Some(context) => {
            crate::indexing::selectors::build_slice_selectors_in_context(
                context,
                dims,
                colon_mask,
                end_mask,
                numeric,
                tensor.shape(),
            )
            .await?
        }
        None => build_slice_selectors(dims, colon_mask, end_mask, numeric, tensor.shape()).await?,
    };
    if let Some(value) =
        try_tensor_slice_2d_fast_path_in_context(context, tensor, dims, &selectors)?
    {
        return Ok(value);
    }
    let plan = match context {
        Some(context) => crate::indexing::plan::build_index_plan_in_context(
            context,
            &selectors,
            dims,
            tensor.shape(),
        )?,
        None => build_index_plan(&selectors, dims, tensor.shape())?,
    };
    payload::real(
        tensor,
        plan.indices.len(),
        |position| plan.indices[position] as usize,
        || Ok(plan.output_shape),
        context,
    )
}

pub fn read_tensor_slice_from_plan(
    tensor: &Tensor,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    read_tensor_slice_from_plan_in_context(context.as_ref(), tensor, plan)
}

pub fn read_tensor_slice_from_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    tensor: &Tensor,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    payload::real(
        tensor,
        plan.indices.len(),
        |position| plan.indices[position] as usize,
        || Ok(plan.output_shape.clone()),
        context,
    )
}

fn sparse_output_shape(plan: &IndexPlan) -> Result<(usize, usize), RuntimeError> {
    match plan.output_shape.as_slice() {
        [rows, cols] => Ok((*rows, *cols)),
        [len] => Ok((*len, 1)),
        _ => Err(crate::runtime_error::semantic_error(
            "UnsupportedSparseIndexRank",
            "Sparse indexing currently supports two-dimensional outputs",
        )),
    }
}

fn checked_sparse_numel(sparse: &SparseTensor) -> Result<usize, RuntimeError> {
    sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
        crate::runtime_error::semantic_error("IndexOutOfBounds", "Sparse dimensions overflow")
    })
}

pub async fn read_sparse_slice(
    sparse: &SparseTensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    read_sparse_slice_in_context(
        context.as_ref(),
        sparse,
        dims,
        colon_mask,
        end_mask,
        numeric,
    )
    .await
}

pub async fn read_sparse_slice_in_context(
    context: Option<&crate::context::RuntimeContext>,
    sparse: &SparseTensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    if sparse.integer_storage().is_some() {
        crate::compatibility::ensure_sparse_integer_extension_enabled("indexed access")?;
    }
    let selectors =
        sparse_slice_selectors(context, sparse, dims, colon_mask, end_mask, numeric).await?;
    sparse_selectors_in_context(
        context,
        sparse,
        dims,
        &selectors,
        SparseReadZeroPolicy::BySelector,
    )
}

/// Select without materializing dense Cartesian coordinates. The explicit
/// zero policy preserves the caller's established physical component contract.
pub async fn read_sparse_slice_with_policy_in_context(
    context: Option<&crate::context::RuntimeContext>,
    sparse: &SparseTensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    policy: SparseReadZeroPolicy,
) -> Result<Value, RuntimeError> {
    let selectors =
        sparse_slice_selectors(context, sparse, dims, colon_mask, end_mask, numeric).await?;
    preserve_sparse_plan_geometry(sparse, policy)?;
    if sparse.integer_storage().is_some() {
        crate::compatibility::ensure_sparse_integer_extension_enabled("indexed access")?;
    }
    sparse_selectors_in_context(context, sparse, dims, &selectors, policy)
}

async fn sparse_slice_selectors(
    context: Option<&crate::context::RuntimeContext>,
    sparse: &SparseTensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<runmat_value::ExactArrayStorage<SliceSelector>, RuntimeError> {
    let shape = [sparse.rows, sparse.cols];
    match context {
        Some(context) => {
            crate::indexing::selectors::build_slice_selectors_in_context(
                context, dims, colon_mask, end_mask, numeric, &shape,
            )
            .await
        }
        None => build_slice_selectors(dims, colon_mask, end_mask, numeric, &shape).await,
    }
}

/// Genuine synchronous host-selector capability. Unsupported selector inputs
/// decline before sparse selection; no future is polled under a native entry.
pub fn try_read_sparse_slice_host_in_context(
    context: &crate::context::RuntimeContext,
    sparse: &SparseTensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    policy: SparseReadZeroPolicy,
) -> Result<Option<Value>, RuntimeError> {
    let shape = [sparse.rows, sparse.cols];
    let Some(selectors) = crate::indexing::selectors::build_slice_selectors_host_in_context(
        context, dims, colon_mask, end_mask, numeric, &shape,
    )?
    else {
        return Ok(None);
    };
    preserve_sparse_plan_geometry(sparse, policy)?;
    if sparse.integer_storage().is_some() {
        crate::compatibility::ensure_sparse_integer_extension_enabled("indexed access")?;
    }
    sparse_selectors_in_context(Some(context), sparse, dims, &selectors, policy).map(Some)
}

fn preserve_sparse_plan_geometry(
    sparse: &SparseTensor,
    policy: SparseReadZeroPolicy,
) -> Result<(), RuntimeError> {
    if policy == SparseReadZeroPolicy::PreserveTypedComponents {
        // Retain the original planned read's checked total geometry without
        // constructing its Cartesian index collection.
        crate::indexing::plan::effective_index_extent(&[sparse.rows, sparse.cols], 1, 0)?;
    }
    Ok(())
}

fn sparse_selectors_in_context(
    context: Option<&crate::context::RuntimeContext>,
    sparse: &SparseTensor,
    dims: usize,
    selectors: &[SliceSelector],
    policy: SparseReadZeroPolicy,
) -> Result<Value, RuntimeError> {
    match dims {
        1 => sparse::linear(
            context,
            sparse,
            selectors
                .first()
                .unwrap_or(&SliceSelector::Indices(Default::default())),
            policy,
        ),
        2 => sparse::matrix(context, sparse, selectors, policy),
        _ => {
            let shape = [sparse.rows, sparse.cols];
            let plan = match context {
                Some(context) => crate::indexing::plan::build_index_plan_in_context(
                    context, selectors, dims, &shape,
                )?,
                None => build_index_plan(selectors, dims, &shape)?,
            };
            sparse::from_plan(context, sparse, &plan)
        }
    }
}

pub fn read_sparse_slice_from_plan(
    sparse: &SparseTensor,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    read_sparse_slice_from_plan_in_context(context.as_ref(), sparse, plan)
}

pub fn read_sparse_slice_from_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    sparse: &SparseTensor,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    sparse::from_plan(context, sparse, plan)
}

pub async fn read_complex_slice(
    tensor: &ComplexTensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    read_complex_slice_in_context(
        context.as_ref(),
        tensor,
        dims,
        colon_mask,
        end_mask,
        numeric,
    )
    .await
}

pub async fn read_complex_slice_in_context(
    context: Option<&crate::context::RuntimeContext>,
    tensor: &ComplexTensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let selectors = match context {
        Some(context) => {
            crate::indexing::selectors::build_slice_selectors_in_context(
                context,
                dims,
                colon_mask,
                end_mask,
                numeric,
                tensor.shape(),
            )
            .await?
        }
        None => build_slice_selectors(dims, colon_mask, end_mask, numeric, tensor.shape()).await?,
    };
    let plan = match context {
        Some(context) => crate::indexing::plan::build_index_plan_in_context(
            context,
            &selectors,
            dims,
            tensor.shape(),
        )?,
        None => build_index_plan(&selectors, dims, tensor.shape())?,
    };
    read_complex_slice_from_plan_in_context(context, tensor, &plan)
}

pub fn read_complex_slice_from_plan(
    tensor: &ComplexTensor,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    read_complex_slice_from_plan_in_context(context.as_ref(), tensor, plan)
}

pub fn read_complex_slice_from_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    tensor: &ComplexTensor,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    payload::complex(tensor, &plan.indices, plan.output_shape.clone(), context)
}

pub async fn read_gpu_slice(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    read_gpu_slice_in_context(
        context.as_ref(),
        handle,
        dims,
        colon_mask,
        end_mask,
        numeric,
    )
    .await
}

pub async fn read_gpu_slice_in_context(
    context: Option<&crate::context::RuntimeContext>,
    handle: &runmat_accelerate_api::GpuTensorHandle,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let base_shape = handle.shape();
    let selectors = match context {
        Some(context) => {
            owner
                .await_stage(
                    crate::indexing::selectors::build_slice_selectors_in_context(
                        context, dims, colon_mask, end_mask, numeric, base_shape,
                    ),
                )
                .await??
        }
        None => build_slice_selectors(dims, colon_mask, end_mask, numeric, base_shape).await?,
    };
    let plan = match context {
        Some(context) => crate::indexing::plan::build_index_plan_in_context(
            context, &selectors, dims, base_shape,
        )?,
        None => build_index_plan(&selectors, dims, base_shape)?,
    };
    read_gpu_slice_from_plan_in_context(context, handle, &plan)
}

pub fn read_gpu_slice_from_plan(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    read_gpu_slice_from_plan_in_context(context.as_ref(), handle, plan)
}

pub fn read_gpu_slice_from_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    handle: &runmat_accelerate_api::GpuTensorHandle,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    super::resident::read_with_plan_in_context(context, handle, plan)
}

pub async fn read_string_slice(
    sa: &StringArray,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let selectors = build_slice_selectors(dims, colon_mask, end_mask, numeric, &sa.shape).await?;
    let plan = build_index_plan(&selectors, dims, &sa.shape)?;
    gather_string_slice(sa, &plan)
}

pub fn gather_string_slice(sa: &StringArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    gather_string_slice_in_context(context.as_ref(), sa, plan)
}
pub fn gather_string_slice_in_context(
    context: Option<&crate::context::RuntimeContext>,
    sa: &StringArray,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    super::value::text::read(context, sa.data(), plan)
}

#[cfg(test)]
mod tests {
    use super::{
        gather_string_slice, read_complex_slice_from_plan, read_gpu_slice_from_plan,
        read_sparse_slice, read_sparse_slice_from_plan, read_string_slice,
        read_tensor_slice_from_plan, try_tensor_slice_2d_fast_path,
    };
    use crate::indexing::plan::IndexPlan;
    use crate::indexing::selectors::SliceSelector;
    use futures::executor::block_on;
    use runmat_value::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, NumericDType,
        NumericScalar, SparseTensor, StringArray, Tensor, Value,
    };

    #[test]
    fn sparse_linear_numeric_vector_selection_uses_source_orientation() {
        let column =
            SparseTensor::new(4, 1, vec![0, 2], vec![1, 2], vec![5.0, 6.0]).expect("column source");
        let row = SparseTensor::new(1, 4, vec![0, 0, 1, 2, 2], vec![0, 0], vec![5.0, 6.0])
            .expect("row source");
        for (source, index_shape, expected) in [
            (&column, vec![1, 2], vec![2, 1]),
            (&row, vec![2, 1], vec![1, 2]),
        ] {
            let selector = Value::Tensor(Tensor::new(vec![2.0, 3.0], index_shape).unwrap());
            let Value::SparseTensor(output) =
                block_on(read_sparse_slice(source, 1, 0, 0, &[selector])).unwrap()
            else {
                panic!("sparse vector result");
            };
            assert_eq!(output.shape(), expected);
            assert_eq!(output.nnz(), 2);
        }
    }

    #[test]
    fn tensor_slice_plan_preserves_exact_uint64_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX, 3, 4]), vec![2, 2])
            .expect("tensor");
        let plan = IndexPlan::new(vec![0, 2, 1, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let result = read_tensor_slice_from_plan(&tensor, &plan).expect("slice");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, 3, u64::MAX, 4]))
        );
    }

    #[test]
    fn sparse_slice_plan_preserves_exact_uint64_storage_and_empty_class() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let sparse = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            IntegerStorage::U64(vec![1, u64::MAX]),
        )
        .expect("sparse");
        let plan = IndexPlan::new(vec![0, 3, 1, 2], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let result = read_sparse_slice_from_plan(&sparse, &plan).expect("slice");
        let Value::SparseTensor(output) = result else {
            panic!("expected sparse output");
        };
        assert_eq!(output.shape(), vec![2, 2]);
        assert_eq!(output.col_ptrs, vec![0, 2, 2]);
        assert_eq!(output.row_indices, vec![0, 1]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, u64::MAX]))
        );

        let empty_plan = IndexPlan::new(Vec::new(), vec![0, 1], vec![0], 1, vec![2, 2]);
        let empty = read_sparse_slice_from_plan(&sparse, &empty_plan).expect("empty slice");
        let Value::SparseTensor(empty) = empty else {
            panic!("expected sparse output");
        };
        assert_eq!(empty.shape(), vec![0, 1]);
        assert_eq!(empty.integer_storage(), Some(&IntegerStorage::U64(vec![])));
    }

    #[test]
    fn sparse_slice_plan_preserves_native_single_storage_and_empty_class() {
        let sparse = SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.25, 3.5])
            .expect("single sparse");
        let plan = IndexPlan::new(vec![0, 3, 1, 2], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let result = read_sparse_slice_from_plan(&sparse, &plan).expect("slice");
        let Value::SparseTensor(output) = result else {
            panic!("expected sparse output");
        };
        assert_eq!(
            output.numeric_dtype(),
            Some(runmat_value::NumericDType::F32)
        );
        assert_eq!(output.col_ptrs, vec![0, 2, 2]);
        assert_eq!(output.row_indices, vec![0, 1]);
        assert_eq!(output.as_f32_slice(), Some(&[1.25, 3.5][..]));

        let empty_plan = IndexPlan::new(Vec::new(), vec![0, 1], vec![0], 1, vec![2, 2]);
        let empty = read_sparse_slice_from_plan(&sparse, &empty_plan).expect("empty slice");
        let Value::SparseTensor(empty) = empty else {
            panic!("expected sparse output");
        };
        assert_eq!(empty.numeric_dtype(), Some(runmat_value::NumericDType::F32));
        assert_eq!(empty.shape(), vec![0, 1]);
        assert_eq!(empty.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn sparse_slice_plan_preserves_logical_pattern_and_empty_class() {
        let sparse =
            SparseTensor::new_logical(2, 2, vec![0, 1, 2], vec![0, 1]).expect("logical sparse");
        let plan = IndexPlan::new(vec![0, 3, 1, 2], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let result = read_sparse_slice_from_plan(&sparse, &plan).expect("slice");
        let Value::SparseTensor(output) = result else {
            panic!("expected sparse output");
        };
        assert!(output.is_logical());
        assert_eq!(output.col_ptrs, vec![0, 2, 2]);
        assert_eq!(output.row_indices, vec![0, 1]);

        let empty_plan = IndexPlan::new(Vec::new(), vec![0, 1], vec![0], 1, vec![2, 2]);
        let empty = read_sparse_slice_from_plan(&sparse, &empty_plan).expect("empty slice");
        let Value::SparseTensor(empty) = empty else {
            panic!("expected sparse output");
        };
        assert!(empty.is_logical());
        assert_eq!(empty.shape(), vec![0, 1]);
        assert_eq!(empty.nnz(), 0);
    }

    #[test]
    fn sparse_slice_plan_preserves_complex_storage_and_empty_class() {
        let sparse = SparseTensor::new_complex(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![(1.0, 2.0), (3.0, -4.0)],
        )
        .expect("complex sparse");
        let plan = IndexPlan::new(vec![0, 3, 1, 2], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let Value::SparseTensor(output) =
            read_sparse_slice_from_plan(&sparse, &plan).expect("slice")
        else {
            panic!("expected sparse output");
        };
        assert!(output.is_complex());
        assert_eq!(
            output.materialize_complex_f64().expect("complex values"),
            vec![(1.0, 2.0), (3.0, -4.0)]
        );

        let empty_plan = IndexPlan::new(Vec::new(), vec![0, 1], vec![0], 1, vec![2, 2]);
        let Value::SparseTensor(empty) =
            read_sparse_slice_from_plan(&sparse, &empty_plan).expect("empty slice")
        else {
            panic!("expected sparse output");
        };
        assert!(empty.is_complex());
        assert_eq!(empty.shape(), vec![0, 1]);
        assert_eq!(empty.nnz(), 0);
    }

    #[test]
    fn sparse_slice_plan_preserves_complex_single_storage() {
        let sparse = SparseTensor::new_complex_f32(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![(1.0, 2.0), (3.0, -4.0)],
        )
        .expect("complex single sparse");
        let plan = IndexPlan::new(vec![0, 3, 1, 2], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let Value::SparseTensor(output) =
            read_sparse_slice_from_plan(&sparse, &plan).expect("slice")
        else {
            panic!("expected sparse output");
        };
        assert!(output.is_complex());
        assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));

        let empty_plan = IndexPlan::new(Vec::new(), vec![0, 1], vec![0], 1, vec![2, 2]);
        let Value::SparseTensor(empty) =
            read_sparse_slice_from_plan(&sparse, &empty_plan).expect("empty slice")
        else {
            panic!("expected sparse output");
        };
        assert!(empty.is_complex());
        assert_eq!(empty.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(empty.shape(), vec![0, 1]);
    }

    #[test]
    fn tensor_slice_fast_path_returns_exact_integer_scalar() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I64(vec![i64::MAX]), vec![1, 1]).expect("tensor");
        let result = try_tensor_slice_2d_fast_path(
            &tensor,
            2,
            &[SliceSelector::Colon, SliceSelector::Scalar(1)],
        )
        .expect("fast path")
        .expect("fast path result");
        assert_eq!(result, Value::Int(IntValue::I64(i64::MAX)));
    }

    #[test]
    fn gpu_integer_slice_preserves_class_for_empty_and_nonempty_plans() {
        runmat_accelerate_api::set_thread_provider(None);
        runmat_accelerate_api::clear_provider();
        runmat_accelerate::simple_provider::register_inprocess_provider();
        let provider = runmat_accelerate_api::provider().expect("test provider");
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

        {
            let source = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[
                        0,
                        1_u64 << 63,
                        u64::MAX,
                    ]),
                    shape: &[1, 3],
                })
                .expect("upload integer gpu source");

            let nonempty = IndexPlan::new(vec![2, 1], vec![1, 2], vec![2], 1, vec![1, 3]);
            let Value::GpuTensor(gathered) =
                read_gpu_slice_from_plan(&source, &nonempty).expect("gpu integer gather")
            else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&gathered),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let host = block_on(provider.download_integer(&gathered)).expect("download gathered");
            assert_eq!(host.shape, vec![1, 2]);
            assert_eq!(
                host.data,
                runmat_accelerate_api::HostIntegerDataOwned::U64(vec![u64::MAX, 1_u64 << 63])
            );

            let empty = IndexPlan::new(Vec::new(), vec![1, 0], vec![0], 1, vec![1, 3]);
            let Value::GpuTensor(empty_handle) =
                read_gpu_slice_from_plan(&source, &empty).expect("empty gpu integer slice")
            else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&empty_handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let host = block_on(provider.download_integer(&empty_handle)).expect("download empty");
            assert_eq!(host.shape, vec![1, 0]);
            assert_eq!(
                host.data,
                runmat_accelerate_api::HostIntegerDataOwned::U64(Vec::new())
            );
        }
    }

    #[test]
    fn string_slice_linear_tensor_indices_preserve_selector_shape() {
        let sa = StringArray::new(
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            vec![2, 2],
        )
        .expect("string array");
        let selector =
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).expect("selector tensor"));
        let result = block_on(read_string_slice(&sa, 1, 0, 0, &[selector])).expect("slice");
        match result {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data(), vec!["a".to_string(), "c".to_string()]);
            }
            other => panic!("expected string array result, got {other:?}"),
        }
    }

    #[test]
    fn string_slice_colon_then_scalar_selects_column() {
        let sa = StringArray::new(
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            vec![2, 2],
        )
        .expect("string array");
        let result =
            block_on(read_string_slice(&sa, 2, 0b01, 0, &[Value::Num(2.0)])).expect("slice");
        match result {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data(), vec!["c".to_string(), "d".to_string()]);
            }
            other => panic!("expected string array result, got {other:?}"),
        }
    }

    #[test]
    fn tensor_slice_plan_preserves_native_single_storage() {
        let tensor = Tensor::from_f32(vec![1.25, 2.5, 3.75], vec![1, 3]).expect("single tensor");
        let plan = IndexPlan::new(vec![2, 0], vec![1, 2], vec![2], 1, vec![1, 3]);
        let Value::Tensor(out) = read_tensor_slice_from_plan(&tensor, &plan).expect("single slice")
        else {
            panic!("expected native-single tensor result");
        };
        assert_eq!(out.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(out.shape(), [1, 2]);
        assert_eq!(out.numeric_value_at(0), Some(NumericScalar::F32(3.75)));
        assert_eq!(out.numeric_value_at(1), Some(NumericScalar::F32(1.25)));
    }

    #[test]
    fn tensor_slice_scalar_preserves_native_single_tensor_class() {
        let tensor = Tensor::from_f32(vec![1.25, 2.5], vec![1, 2]).expect("single tensor");
        let plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 2]);
        let Value::Tensor(out) =
            read_tensor_slice_from_plan(&tensor, &plan).expect("single scalar slice")
        else {
            panic!("expected native-single scalar tensor");
        };
        assert_eq!(out.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(out.numeric_value_at(0), Some(NumericScalar::F32(2.5)));
    }

    #[test]
    fn tensor_slice_plan_shape_mismatch_reports_identifier() {
        let tensor = Tensor::new(vec![10.0, 20.0], vec![1, 2]).expect("tensor");
        let plan = IndexPlan::new(vec![0, 1], vec![1, 1], vec![2], 1, vec![1, 2]);
        let err = read_tensor_slice_from_plan(&tensor, &plan)
            .expect_err("shape-mismatch plan should fail");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn string_slice_plan_shape_mismatch_reports_identifier() {
        let sa = StringArray::new(
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            vec![2, 2],
        )
        .expect("string array");
        let plan = IndexPlan::new(vec![0, 1], vec![1, 1], vec![2], 1, vec![2, 2]);
        let err = gather_string_slice(&sa, &plan).expect_err("shape-mismatch plan should fail");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn complex_slice_plan_shape_mismatch_reports_identifier() {
        let ct = ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0)], vec![1, 2]).expect("complex");
        let plan = IndexPlan::new(vec![0, 1], vec![1, 1], vec![2], 1, vec![1, 2]);
        let err =
            read_complex_slice_from_plan(&ct, &plan).expect_err("shape-mismatch plan should fail");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn integer_complex_slice_preserves_exact_reordered_and_empty_components() {
        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![1, 9_223_372_036_854_775_809, u64::MAX]),
                IntegerStorage::U64(vec![7, 8, 9]),
            )
            .unwrap(),
            vec![1, 3],
        )
        .unwrap();
        let reordered = IndexPlan::new(vec![2, 0], vec![1, 2], vec![2], 1, vec![1, 3]);
        let Value::ComplexTensor(result) =
            read_complex_slice_from_plan(&complex, &reordered).expect("typed complex slice")
        else {
            panic!("typed complex selection must remain a complex tensor");
        };
        assert_eq!(result.shape(), [1, 2]);
        assert_eq!(
            result.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX, 1]),
                    IntegerStorage::U64(vec![9, 7]),
                )
                .unwrap()
            )
        );

        let scalar = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 3]);
        let Value::ComplexTensor(result) =
            read_complex_slice_from_plan(&complex, &scalar).expect("typed scalar selection")
        else {
            panic!("typed complex scalar must retain exact complex storage");
        };
        assert_eq!(
            result.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![9_223_372_036_854_775_809]),
                    IntegerStorage::U64(vec![8]),
                )
                .unwrap()
            )
        );

        let empty = IndexPlan::new(Vec::new(), vec![0, 1], vec![0], 1, vec![1, 3]);
        let Value::ComplexTensor(result) =
            read_complex_slice_from_plan(&complex, &empty).expect("empty typed selection")
        else {
            panic!("empty typed complex selection must retain its class");
        };
        assert_eq!(
            result.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(Vec::new()),
                    IntegerStorage::U64(Vec::new())
                )
                .unwrap()
            )
        );
    }

    #[test]
    fn slice_acceleration_error_mapping_reports_identifier() {
        let err =
            crate::indexing::resident::failure("slice", anyhow::anyhow!("provider failed").into());
        assert_eq!(err.identifier(), Some("RunMat:AccelerationOperationFailed"));
    }
}
