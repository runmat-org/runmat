use crate::builtins::common::tensor::tensor_element_len;
use crate::indexing::integer_assignment;
use crate::indexing::plan::IndexPlan;
use crate::runtime_error::semantic_error as mex;
use crate::RuntimeError;
use runmat_value::{
    ComplexTensor, NumericDType, NumericScalar, NumericStorage, SparseTensor, StringArray, Tensor,
    Value,
};

fn map_slice_shape_error(context: &str, err: impl std::fmt::Display) -> RuntimeError {
    mex("ShapeMismatch", format!("{context}: {err}"))
}

fn dense_delete_dimensions(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (0, 0),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    }
}

fn is_empty_delete_rhs(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(t)
            if {
                let (rows, cols) = dense_delete_dimensions(t.shape());
                tensor_element_len(t) == 0 || rows == 0 || cols == 0
            }
    ) || matches!(
        value,
        Value::ComplexTensor(t)
            if t.is_empty() || t.rows() == 0 || t.cols() == 0
    )
}

fn deleted_vector_dimensions(rows: usize, len: usize) -> [usize; 2] {
    if len == 0 {
        [0, 0]
    } else if rows == 1 {
        [1, len]
    } else {
        [len, 1]
    }
}

fn validated_delete_position(index: u32, total: usize) -> Result<usize, RuntimeError> {
    usize::try_from(index)
        .ok()
        .filter(|position| *position < total)
        .ok_or_else(|| mex("IndexOutOfBounds", "Index out of bounds"))
}

/// Canonicalize already validated scratch in place; the returned prefix is
/// unique and descending. Both numeric and logical deletion use this owner.
fn canonicalize_delete_positions_desc(positions: &mut [usize]) -> usize {
    positions.sort_unstable();
    let mut unique = 0;
    for index in 0..positions.len() {
        if unique == 0 || positions[index] != positions[unique - 1] {
            positions[unique] = positions[index];
            unique += 1;
        }
    }
    positions[..unique].reverse();
    unique
}

mod empty_rhs;
mod logical_delete;
pub use logical_delete::{delete_bool_with_plan, delete_logical_with_plan};
mod numeric_delete;
pub(in crate::indexing) use numeric_delete::{
    delete_complex_at_index_in_context, delete_tensor_at_index_in_context,
};
pub use numeric_delete::{delete_complex_with_plan_in_context, delete_tensor_with_plan_in_context};
mod dense_assignment;
mod growth;
mod host_rhs;
pub(in crate::indexing) mod rhs_shape;
pub(in crate::indexing) mod sparse;
pub use dense_assignment::{
    assign_complex_with_plan_host_in_context, assign_tensor_with_plan_host_in_context,
};
pub(crate) use rhs_shape::AssignmentRhsLayout;
use rhs_shape::{checked_assignment_element_count, validated_assignment_rhs_shape};

/// Validated borrowed native complex interpretation; no payload/stride mirror.
pub struct ComplexRhsView<'a>(host_rhs::HostRhs<'a>);
impl ComplexRhsView<'_> {
    fn tensor_value(&self, ordinal: usize) -> (f64, f64) {
        self.0.complex_at(ordinal)
    }
}
pub fn build_complex_rhs_view<'a>(
    rhs: &'a Value,
    selection_lengths: &[usize],
) -> Result<ComplexRhsView<'a>, RuntimeError> {
    host_rhs::HostRhs::complex(rhs, selection_lengths, false).map(ComplexRhsView)
}
pub fn scatter_complex_with_plan(
    tensor: &mut ComplexTensor,
    plan: &IndexPlan,
    rhs: &ComplexRhsView<'_>,
) -> Result<(), RuntimeError> {
    if rhs.0.len() != plan.indices.len() {
        return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
    }
    for (ordinal, &destination) in plan.indices.iter().enumerate() {
        let (real, imag) = rhs.tensor_value(ordinal);
        tensor
            .set_f64_assignment_at(destination as usize, real, imag)
            .map_err(|error| map_slice_shape_error("complex slice assign", error))?;
    }
    Ok(())
}

pub use super::value::text::StringRhsView;
mod logical_assignment;
pub use logical_assignment::{
    assign_logical_with_plan_host_in_context, assign_logical_with_plan_in_context,
};

pub fn validate_string_rhs_for_plan(
    rhs: &Value,
    selection_lengths: &[usize],
) -> Result<(), RuntimeError> {
    super::value::text::validate_rhs(rhs, selection_lengths)
}
pub fn build_string_rhs_view(
    rhs: &Value,
    selection_lengths: &[usize],
) -> Result<StringRhsView, RuntimeError> {
    let context = crate::context::legacy::active();
    super::value::text::build_rhs(context.as_ref(), rhs, selection_lengths)
}
pub fn scatter_string_with_plan(
    value: &mut StringArray,
    plan: &IndexPlan,
    view: &StringRhsView,
) -> Result<(), RuntimeError> {
    let context = crate::context::legacy::active();
    super::value::text::scatter_existing(context.as_ref(), value, plan, view)
}
pub fn assign_string_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    value: StringArray,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    super::value::text::assign(context, value, plan, rhs)
}

pub async fn materialize_rhs_real_for_plan(
    rhs: &Value,
    plan: &IndexPlan,
) -> Result<Vec<f64>, RuntimeError> {
    let host_rhs = crate::dispatcher::gather_if_needed_async(rhs).await?;
    materialize_rhs_real_for_plan_host(&host_rhs, plan)
}

pub fn materialize_rhs_real_for_plan_host(
    rhs: &Value,
    plan: &IndexPlan,
) -> Result<Vec<f64>, RuntimeError> {
    if plan.dims == 1 {
        let count = plan.selection_lengths.first().copied().unwrap_or(0);
        materialize_rhs_linear_real_host(rhs, count)
    } else {
        materialize_rhs_nd_real_host(rhs, &plan.selection_lengths)
    }
}

pub fn scatter_real_with_plan(
    storage: &mut NumericStorage,
    plan: &IndexPlan,
    rhs_values: &[f64],
) -> Result<(), RuntimeError> {
    scatter_real_values(storage, plan, rhs_values.len(), |ordinal| {
        rhs_values[ordinal]
    })
}

fn scatter_real_values(
    storage: &mut NumericStorage,
    plan: &IndexPlan,
    rhs_len: usize,
    mut value_at: impl FnMut(usize) -> f64,
) -> Result<(), RuntimeError> {
    if rhs_len != plan.indices.len() {
        return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
    }
    for (ordinal, &dst) in plan.indices.iter().enumerate() {
        let value = value_at(ordinal);
        let value = match storage.numeric_dtype() {
            NumericDType::F64 => NumericScalar::F64(value),
            NumericDType::F32 => NumericScalar::F32(value as f32),
            _ => unreachable!("real floating scatter requires floating storage"),
        };
        storage
            .set_value(dst as usize, value)
            .map_err(|error| map_slice_shape_error("slice assign", error))?;
    }
    Ok(())
}

/// Assigns an index-plan selection into a host-resident CSC sparse matrix.
/// Sparse storage owns the batched merge; this module owns MATLAB selector and
/// RHS assignment-shape semantics shared with dense tensor assignment.
pub async fn assign_sparse_with_plan(
    source: SparseTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    assign_sparse_with_plan_in_context(context.as_ref(), source, plan, rhs).await
}

pub async fn assign_sparse_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    source: SparseTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    sparse::assign(context, source, plan, rhs).await
}

pub(in crate::indexing) async fn gather_assignment_rhs_in_context(
    context: Option<&crate::context::RuntimeContext>,
    rhs: &Value,
    integer_target: bool,
) -> Result<Value, RuntimeError> {
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
    };
    let policy = HostAllocationReceipt::policy(context).map_err(|source| {
        host_allocation_runtime_error(
            context,
            "RunMat:Indexing:PayloadAllocation",
            "assignment RHS download",
            source,
        )
    })?;
    if policy == HostAllocationPolicy::Managed {
        // The existing exact native download/gather owner retains physical
        // callback claims independently of this logical cancellation waiter.
        let host = super::resident::gather_dispatch_value_in_context(rhs, context).await?;
        if let Value::GpuTensor(handle) = rhs {
            if integer_target && runmat_accelerate_api::handle_integer_type(handle).is_some() {
                return match host {
                    Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
                        Ok(Value::Tensor(tensor))
                    }
                    _ => Err(mex(
                        "AccelerationOperationFailed",
                        "Expected a native real integer payload",
                    )),
                };
            }
        }
        return Ok(host);
    }
    if let Value::GpuTensor(handle) = rhs {
        if integer_target && runmat_accelerate_api::handle_integer_type(handle).is_some() {
            return download_integer_tensor_in_context(context, handle)
                .await
                .map(Value::Tensor);
        }
    }
    super::resident::gather_dispatch_value_in_context(rhs, context).await
}

fn preflight_sparse_assignment(
    sparse: &SparseTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<(), RuntimeError> {
    if sparse.integer_storage().is_some() {
        crate::compatibility::ensure_sparse_integer_extension_enabled("indexed assignment")?;
    }
    // Deletion is a distinct compiler-selected operation. A computed empty RHS
    // may still assign to an empty selection when its shape is compatible.
    if !plan.indices.is_empty() && is_empty_delete_rhs(rhs) {
        return Err(mex(
            "SparseAssignmentUnsupported",
            "Sparse indexed deletion is not yet supported",
        ));
    }
    if plan.indices.is_empty() {
        empty_rhs::validate_sparse_rhs(sparse, rhs, plan)?;
    }
    Ok(())
}

pub fn assign_sparse_with_plan_host(
    source: SparseTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    assign_sparse_with_plan_host_in_context(context.as_ref(), source, plan, rhs)
}

pub fn assign_sparse_with_plan_host_in_context(
    context: Option<&crate::context::RuntimeContext>,
    source: SparseTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    sparse::assign_host(context, source, plan, rhs)
}

pub fn delete_sparse_with_plan(
    source: SparseTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    delete_sparse_with_plan_in_context(context.as_ref(), source, plan, rhs)
}

pub fn delete_sparse_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    source: SparseTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    sparse::delete(context, source, plan, rhs)
}

pub async fn assign_tensor_with_plan(
    t: Tensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    assign_tensor_with_plan_in_context(context.as_ref(), t, plan, rhs).await
}

pub async fn assign_tensor_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    t: Tensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    if !matches!(rhs, Value::GpuTensor(_)) || plan.indices.is_empty() {
        return assign_tensor_with_plan_host_in_context(context, t, plan, rhs);
    }
    preflight_tensor_assignment(&t, plan, rhs)?;
    let host_rhs = owner
        .await_stage(gather_assignment_rhs_in_context(
            context,
            rhs,
            t.integer_storage().is_some(),
        ))
        .await??;
    if matches!(host_rhs, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        // Resident RHS gathering did not historically authorize real-target
        // complex promotion; that route is selected by a host complex RHS.
        host_rhs::HostRhs::for_plan(&host_rhs, plan)?;
    }
    owner.check()?;
    assign_tensor_with_plan_host_in_context(context, t, plan, &host_rhs)
}

fn preflight_tensor_assignment(
    t: &Tensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<(), RuntimeError> {
    if plan.indices.is_empty() {
        empty_rhs::validate_real_or_host_complex_rhs(rhs, plan)?;
        return Ok(());
    }
    if matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        t.validate_shape()
            .map_err(|error| map_slice_shape_error("slice complex promotion", error))?;
        return Ok(());
    }
    let target_len = plan.base_shape.iter().try_fold(1usize, |len, extent| {
        len.checked_mul(*extent).ok_or_else(|| {
            mex(
                "ShapeOverflow",
                "slice assignment target shape overflows usize",
            )
        })
    })?;
    t.validate_shape()
        .map_err(|error| map_slice_shape_error("slice assign", error))?;
    if target_len < t.len() {
        return Err(mex(
            "InvalidAssignmentPlan",
            "slice assignment plan cannot shrink numeric storage",
        ));
    }
    Ok(())
}

pub fn assign_tensor_with_plan_host(
    t: Tensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    assign_tensor_with_plan_host_in_context(context.as_ref(), t, plan, rhs)
}

pub async fn assign_complex_with_plan(
    tensor: ComplexTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    assign_complex_with_plan_in_context(context.as_ref(), tensor, plan, rhs).await
}

pub async fn assign_complex_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    tensor: ComplexTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    if !matches!(rhs, Value::GpuTensor(_))
        || plan.indices.is_empty()
        || tensor.integer_storage().is_none()
    {
        return assign_complex_with_plan_host_in_context(context, tensor, plan, rhs);
    }
    let host_rhs = owner
        .await_stage(gather_assignment_rhs_in_context(context, rhs, true))
        .await??;
    owner.check()?;
    assign_complex_with_plan_host_in_context(context, tensor, plan, &host_rhs)
}

pub fn assign_complex_with_plan_host(
    tensor: ComplexTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    assign_complex_with_plan_host_in_context(context.as_ref(), tensor, plan, rhs)
}

pub fn delete_tensor_with_plan(
    t: Tensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    delete_tensor_with_plan_in_context(context.as_ref(), t, plan, rhs)
}

pub fn delete_complex_with_plan(
    t: ComplexTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    delete_complex_with_plan_in_context(context.as_ref(), t, plan, rhs)
}

mod resident;
pub use resident::{
    assign_gpu_slice_with_plan, assign_gpu_slice_with_plan_in_context, delete_gpu_slice_with_plan,
    delete_gpu_slice_with_plan_in_context,
};

fn integer_gpu_rhs_indices_for_plan(
    rhs_shape: &[usize],
    plan: &IndexPlan,
) -> Result<Vec<u32>, RuntimeError> {
    let rhs_len = rhs_shape.iter().try_fold(1usize, |len, dimension| {
        len.checked_mul(*dimension)
            .ok_or_else(|| mex("ShapeMismatch", "shape mismatch for slice assign"))
    })?;
    if plan.dims == 1 {
        if rhs_len == plan.indices.len() {
            return (0..rhs_len)
                .map(|index| {
                    u32::try_from(index).map_err(|_| {
                        mex(
                            "AccelerationOperationFailed",
                            "GPU rhs exceeds indexing limits",
                        )
                    })
                })
                .collect();
        }
        if rhs_len == 1 {
            return Ok(vec![0; plan.indices.len()]);
        }
        return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
    }

    let dims = plan.selection_lengths.len();
    let shape = validated_assignment_rhs_shape(rhs_shape, rhs_len, &plan.selection_lengths)?;
    let mut strides = vec![1usize; dims];
    for dimension in 1..dims {
        strides[dimension] = strides[dimension - 1]
            .checked_mul(shape[dimension - 1].max(1))
            .ok_or_else(|| mex("ShapeMismatch", "shape mismatch for slice assign"))?;
    }
    let mut output = Vec::with_capacity(plan.indices.len());
    let mut coordinates = vec![0usize; dims];
    for _ in 0..plan.indices.len() {
        let mut rhs_index = 0usize;
        for dimension in 0..dims {
            let coordinate = if shape[dimension] == 1 {
                0
            } else {
                coordinates[dimension]
            };
            rhs_index = rhs_index
                .checked_add(
                    coordinate
                        .checked_mul(strides[dimension])
                        .ok_or_else(|| mex("ShapeMismatch", "shape mismatch for slice assign"))?,
                )
                .ok_or_else(|| mex("ShapeMismatch", "shape mismatch for slice assign"))?;
        }
        output.push(u32::try_from(rhs_index).map_err(|_| {
            mex(
                "AccelerationOperationFailed",
                "GPU rhs exceeds indexing limits",
            )
        })?);
        for (dimension, coordinate) in coordinates.iter_mut().enumerate().take(dims) {
            *coordinate += 1;
            if *coordinate < plan.selection_lengths[dimension].max(1) {
                break;
            }
            *coordinate = 0;
        }
    }
    Ok(output)
}

pub async fn materialize_rhs_linear_real(
    rhs: &Value,
    count: usize,
) -> Result<Vec<f64>, RuntimeError> {
    let host_rhs = crate::dispatcher::gather_if_needed_async(rhs).await?;
    materialize_rhs_linear_real_host(&host_rhs, count)
}

pub fn materialize_rhs_linear_real_host(
    rhs: &Value,
    count: usize,
) -> Result<Vec<f64>, RuntimeError> {
    host_rhs::linear(rhs, count)
}

pub async fn materialize_rhs_nd_real(
    rhs: &Value,
    selection_lengths: &[usize],
) -> Result<Vec<f64>, RuntimeError> {
    let rhs_host = crate::dispatcher::gather_if_needed_async(rhs).await?;
    materialize_rhs_nd_real_host(&rhs_host, selection_lengths)
}

pub fn materialize_rhs_nd_real_host(
    rhs: &Value,
    selection_lengths: &[usize],
) -> Result<Vec<f64>, RuntimeError> {
    host_rhs::multidimensional(rhs, selection_lengths)
}

pub(crate) async fn download_integer_tensor_in_context(
    context: Option<&crate::context::RuntimeContext>,
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> Result<Tensor, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    match owner
        .await_stage(super::resident::gather_in_context(context, handle))
        .await??
    {
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => Ok(tensor),
        _ => Err(mex(
            "AccelerationOperationFailed",
            "Expected a native real integer payload",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        assign_complex_with_plan, assign_gpu_slice_with_plan, assign_sparse_with_plan,
        assign_tensor_with_plan, build_complex_rhs_view, build_string_rhs_view,
        delete_complex_with_plan, delete_gpu_slice_with_plan, delete_sparse_with_plan,
        delete_tensor_with_plan, integer_gpu_rhs_indices_for_plan, materialize_rhs_linear_real,
        materialize_rhs_nd_real, validated_assignment_rhs_shape,
    };
    use crate::indexing::plan::{build_assignment_plan, build_sparse_assignment_plan, IndexPlan};
    use crate::indexing::selectors::SliceSelector;
    use futures::executor::block_on;
    use runmat_value::{
        CellArray, ComplexTensor, IntegerComplexStorage, IntegerStorage, NumericDType,
        NumericStorage, SparseTensor, StringArray, Tensor, Value,
    };

    #[test]
    fn gpu_assignment_rejects_repeated_target_subscripts_before_provider_access() {
        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 3],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered input")
            .handle;
        let plan = IndexPlan::new(vec![1, 1], vec![1, 2], vec![2], 1, vec![1, 3]);
        let error = block_on(assign_gpu_slice_with_plan(&handle, &plan, &Value::Num(4.0)))
            .expect_err("repeated gpuArray assignment targets must reject");
        assert_eq!(
            error.identifier(),
            Some("RunMat:RepeatedGpuAssignmentIndex")
        );
    }

    #[test]
    fn integer_plan_assignment_preserves_exact_uint64_rhs() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 3]), vec![1, 3]).expect("tensor");
        let rhs_tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 9]), vec![1, 2]).expect("rhs");
        let rhs = Value::Tensor(rhs_tensor);
        let plan = IndexPlan::new(vec![0, 1], vec![1, 2], vec![2], 1, vec![1, 3]);
        let result = block_on(assign_tensor_with_plan(tensor, &plan, &rhs)).expect("assign");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 9, 3]))
        );
    }

    #[test]
    fn native_single_plan_assignment_and_deletion_preserve_class() {
        let tensor = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("single tensor");
        let plan = IndexPlan::new(vec![0, 2], vec![1, 2], vec![2], 1, vec![1, 3]);
        let Value::Tensor(updated) = block_on(assign_tensor_with_plan(
            tensor,
            &plan,
            &Value::Num(1.234_567_890_123),
        ))
        .expect("single slice assignment") else {
            panic!("expected tensor");
        };
        assert_eq!(updated.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            updated.clone().into_numeric_storage(),
            Ok(NumericStorage::F32(vec![
                1.234_567_890_123_f64 as f32,
                2.0,
                1.234_567_890_123_f64 as f32,
            ]))
        );

        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"));
        let delete_plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 3]);
        let Value::Tensor(deleted) =
            delete_tensor_with_plan(updated, &delete_plan, &empty).expect("single slice deletion")
        else {
            panic!("expected tensor");
        };
        assert_eq!(deleted.numeric_dtype(), NumericDType::F32);
        assert_eq!(deleted.shape(), [1, 2]);
        assert_eq!(
            deleted.into_numeric_storage(),
            Ok(NumericStorage::F32(vec![
                1.234_567_890_123_f64 as f32,
                1.234_567_890_123_f64 as f32,
            ]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn rank_zero_and_rank_one_plan_deletion_keep_dense_vector_geometry() {
        let rank_zero = Tensor::new(vec![7.0], vec![]).expect("rank-zero tensor");
        let rank_zero_plan = IndexPlan::new(vec![0], vec![1], vec![1], 1, vec![]);
        let rejected = delete_tensor_with_plan(
            rank_zero.clone(),
            &rank_zero_plan,
            &Value::Tensor(rank_zero.clone()),
        )
        .expect_err("rank-zero cached geometry is not a vector");
        assert_eq!(rejected.identifier(), Some("RunMat:UnsupportedDeletion"));

        let rank_one = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).expect("rank-one tensor");
        let rank_one_plan = IndexPlan::new(vec![1], vec![1], vec![1], 1, vec![3]);
        let Value::Tensor(deleted) =
            delete_tensor_with_plan(rank_one, &rank_one_plan, &Value::Tensor(rank_zero))
                .expect("rank-zero RHS keeps cached-empty deletion admission")
        else {
            panic!("expected tensor");
        };
        assert_eq!(deleted.shape(), [1, 2]);
        assert_eq!(deleted.materialize_f64(), vec![1.0, 3.0]);

        let complex_rank_zero =
            ComplexTensor::new(vec![(7.0, -7.0)], vec![]).expect("rank-zero complex tensor");
        let empty = Value::Tensor(Tensor::zeros(vec![0, 0]));
        let rejected = delete_complex_with_plan(complex_rank_zero, &rank_zero_plan, &empty)
            .expect_err("rank-zero complex geometry is not a vector");
        assert_eq!(rejected.identifier(), Some("RunMat:UnsupportedDeletion"));

        let complex_rank_one =
            ComplexTensor::new(vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0)], vec![3])
                .expect("rank-one complex tensor");
        let Value::ComplexTensor(deleted) =
            delete_complex_with_plan(complex_rank_one, &rank_one_plan, &empty)
                .expect("rank-one complex deletion")
        else {
            panic!("expected complex tensor");
        };
        assert_eq!(deleted.shape(), [1, 2]);
        assert_eq!(deleted.materialize_f64(), vec![(1.0, -1.0), (3.0, -3.0)]);

        // The same deletion geometry/position owner keeps logical bytes in
        // their native class, including duplicate selectors and orientation.
        use runmat_value::LogicalArray;
        for (shape, expected_shape) in [
            (vec![1, 4], vec![1, 2]),
            (vec![4, 1], vec![2, 1]),
            (vec![4], vec![1, 2]),
        ] {
            let mut source = LogicalArray::new(vec![1, 0, 1, 0], shape.clone()).unwrap();
            source.data[0] = 2; // Host mutation still has nonzero logical semantics.
            let plan = IndexPlan::new(vec![3, 1, 1], vec![1, 3], vec![3], 1, shape.clone());
            let real = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], shape.clone()).unwrap();
            let alias = real.clone();
            let Value::Tensor(output) = delete_tensor_with_plan(real, &plan, &empty).unwrap()
            else {
                panic!("deletion changed tensor class")
            };
            assert_eq!(output.shape(), expected_shape);
            assert_eq!(output.as_f32_slice().unwrap(), &[1.0, 3.0]);
            assert_eq!(alias.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
            let complex = ComplexTensor::from_f64_values_with_dtype(
                vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)],
                shape.clone(),
                NumericDType::F32,
            )
            .unwrap();
            let Value::ComplexTensor(output) =
                delete_complex_with_plan(complex, &plan, &empty).unwrap()
            else {
                panic!("deletion changed complex tensor class")
            };
            assert_eq!(output.shape(), expected_shape);
            assert_eq!(output.numeric_dtype(), NumericDType::F32);
            assert_eq!(output.materialize_f64(), vec![(1.0, -1.0), (3.0, -3.0)]);
            let all = IndexPlan::new(vec![0, 1, 2, 3], vec![1, 4], vec![4], 1, shape.clone());
            let Value::Tensor(output) = delete_tensor_with_plan(alias, &all, &empty).unwrap()
            else {
                panic!("empty deletion result must remain a tensor")
            };
            assert_eq!(output.shape(), [0, 0]);
            assert_eq!(output.numeric_dtype(), NumericDType::F32);
            let Value::LogicalArray(output) = block_on(super::delete_logical_with_plan(
                None,
                source.clone(),
                &plan,
                &empty,
            ))
            .unwrap() else {
                panic!("logical deletion changed class")
            };
            assert_eq!(output.data.as_slice(), &[1, 1]);
            assert_eq!(output.shape.as_slice(), expected_shape);
            assert_eq!(source.data.as_slice(), &[2, 0, 1, 0]);
        }
        let source = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let all = IndexPlan::new(vec![1, 0], vec![1, 2], vec![2], 1, vec![1, 2]);
        let Value::LogicalArray(output) = block_on(super::delete_logical_with_plan(
            None,
            source.clone(),
            &all,
            &empty,
        ))
        .unwrap() else {
            panic!("logical deletion changed class")
        };
        assert!(output.data.is_empty());
        assert_eq!(output.shape.as_slice(), &[0, 0]);
        let invalid = IndexPlan::new(vec![2], vec![1, 1], vec![1], 1, vec![1, 2]);
        assert!(block_on(super::delete_logical_with_plan(
            None,
            source.clone(),
            &invalid,
            &empty
        ))
        .is_err());
        assert!(block_on(super::delete_logical_with_plan(
            None,
            source.clone(),
            &all,
            &Value::Bool(false)
        ))
        .is_err());
        assert!(block_on(super::delete_logical_with_plan(
            None,
            LogicalArray::new(vec![1; 4], vec![2, 2]).unwrap(),
            &all,
            &empty
        ))
        .is_err());

        // Fixture inputs are ordinary owners. The operation itself is finite:
        // retained logical payload/shape claims survive scratch retirement.
        use crate::context::{
            host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
            RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
        };
        use std::rc::Rc;
        struct Managed(ManagedHostEffectLedger);
        impl RuntimeParallelService for Managed {
            fn supports(&self, _: ParallelCapability) -> bool {
                false
            }
            fn placement_resources(&self) -> RuntimeParallelResources {
                let snapshot = self.0.snapshot();
                RuntimeParallelResources {
                    memory_available_bytes: Some(snapshot.free_bytes()),
                    epoch: snapshot.epoch,
                    ..Default::default()
                }
            }
            fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
                Some(self.0.clone())
            }
        }
        let managed = |capacity| {
            let ledger = ManagedHostEffectLedger::new_with_epoch(capacity, 309);
            let runtime =
                RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                    .with_service_ports(
                        RuntimeServicePorts::default()
                            .with_parallel(Rc::new(Managed(ledger.clone()))),
                    );
            (runtime, ledger)
        };
        let (runtime, ledger) = managed(1_000_000);
        let output = block_on(super::delete_logical_with_plan(
            Some(&runtime),
            source.clone(),
            &all,
            &empty,
        ))
        .unwrap();
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        assert_eq!(
            ledger.snapshot().live_reserved_bytes,
            ledger.snapshot().live_claimed_bytes
        );
        drop(output);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        let (denied, ledger) = managed(0);
        assert!(block_on(super::delete_logical_with_plan(
            Some(&denied),
            source.clone(),
            &all,
            &empty
        ))
        .is_err());
        let no_indices = IndexPlan::new(Vec::new(), vec![1, 0], vec![0], 1, vec![1, 2]);
        let Value::LogicalArray(unchanged) = block_on(super::delete_logical_with_plan(
            Some(&denied),
            source.clone(),
            &no_indices,
            &empty,
        ))
        .unwrap() else {
            panic!("logical no-op changed class")
        };
        assert!(unchanged.data.shares_allocation_with(&source.data));
        // The scalar no-op remains inline, with no one-element array to fund.
        assert_eq!(
            block_on(super::delete_bool_with_plan(
                Some(&denied),
                true,
                &no_indices,
                &empty
            ))
            .unwrap(),
            Value::Bool(true)
        );
        let scalar_delete = IndexPlan::new(vec![0], vec![1, 1], vec![1], 1, vec![1, 1]);
        assert!(block_on(super::delete_bool_with_plan(
            Some(&denied),
            true,
            &scalar_delete,
            &empty
        ))
        .is_err());
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        let (cancelled, ledger) = managed(1_000_000);
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let cancelled = cancelled.with_cancellation_token(cancellation.token());
        assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
        assert!(cancelled.is_cancelled());
        let error = block_on(super::delete_logical_with_plan(
            Some(&cancelled),
            source.clone(),
            &all,
            &empty,
        ))
        .expect_err("stopped logical deletion must not publish");
        assert!(matches!(
            std::error::Error::source(&error)
                .and_then(|cause| cause.downcast_ref::<crate::execution::ExecutionServiceError>()),
            Some(crate::execution::ExecutionServiceError::Cancelled)
        ));
        assert_eq!(source.data.as_slice(), &[1, 0]);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        // The same managed fixture also qualifies the reached primitive
        // read/write owners, with no ambient activation or payload mirrors.
        use runmat_value::{CharArray, Utf16CodeUnit};
        let (runtime, ledger) = managed(1_000_000);
        let select = IndexPlan::new(vec![1, 2], vec![1, 2], vec![2], 1, vec![2, 2]);
        let chars = || {
            CharArray::from_code_units_with_shape(
                vec![
                    Utf16CodeUnit::from_raw(0xd800),
                    Utf16CodeUnit::from_raw(0),
                    Utf16CodeUnit::from_raw(0xdc00),
                    Utf16CodeUnit::from_raw(65),
                ],
                vec![2, 2],
            )
            .unwrap()
        };
        for base in [
            Value::LogicalArray(LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap()),
            Value::CharArray(chars()),
            Value::StringArray(
                StringArray::new(
                    vec!["a".into(), "b".into(), "c".into(), "d".into()],
                    vec![2, 2],
                )
                .unwrap(),
            ),
        ] {
            let selected =
                crate::indexing::value::read_with_plan_in_context(Some(&runtime), base, &select)
                    .unwrap();
            match &selected {
                Value::LogicalArray(value) => assert_eq!(value.data.as_slice(), &[0, 1]),
                Value::CharArray(value) => assert_eq!(
                    value
                        .iter_column_major()
                        .map(Utf16CodeUnit::raw)
                        .collect::<Vec<_>>(),
                    [0xdc00, 0]
                ),
                Value::StringArray(value) => assert_eq!(value.data(), ["b", "c"]),
                _ => panic!("primitive read changed class"),
            }
            assert!(ledger.snapshot().live_claimed_bytes > 0);
            let (pointer, rhs) = match &selected {
                Value::LogicalArray(value) => (value.data.as_ptr() as usize, Value::Bool(true)),
                Value::CharArray(value) => (
                    value.data().as_ptr() as usize,
                    Value::CharArray(CharArray::new_row("\0")),
                ),
                Value::StringArray(value) => (
                    value.data().as_ptr() as usize,
                    Value::String(Default::default()),
                ),
                _ => unreachable!("primitive class checked above"),
            };
            let before = ledger.snapshot().live_claimed_bytes;
            let store = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 2]);
            let updated = block_on(crate::indexing::value::assign_with_plan_in_context(
                Some(&runtime),
                selected,
                &store,
                rhs,
            ))
            .unwrap();
            let after = match &updated {
                Value::LogicalArray(value) => value.data.as_ptr() as usize,
                Value::CharArray(value) => value.data().as_ptr() as usize,
                Value::StringArray(value) => value.data().as_ptr() as usize,
                _ => panic!("primitive unique class"),
            };
            assert_eq!(
                after, pointer,
                "admitted unique updates retain actual backing"
            );
            assert!(ledger.snapshot().live_claimed_bytes <= before);
            drop(updated);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
        let update = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 3]);
        let (zero, ledger) = managed(0);
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let pointer = logical.data.as_ptr();
        let Value::LogicalArray(updated) =
            block_on(crate::indexing::value::assign_with_plan_in_context(
                Some(&zero),
                Value::LogicalArray(logical),
                &update,
                Value::Bool(true),
            ))
            .unwrap()
        else {
            panic!("logical unique class")
        };
        assert_eq!(updated.data.as_ptr(), pointer);
        assert_eq!(updated.data.as_slice(), &[1, 1, 1]);
        drop(updated);
        let character = CharArray::new_row("abc");
        let pointer = character.data().as_ptr();
        let Value::CharArray(updated) =
            block_on(crate::indexing::value::assign_with_plan_in_context(
                Some(&zero),
                Value::CharArray(character),
                &update,
                Value::CharArray(CharArray::new_row("\0")),
            ))
            .unwrap()
        else {
            panic!("character unique class")
        };
        assert_eq!(updated.data().as_ptr(), pointer);
        assert_eq!(updated.data()[1].raw(), 0);
        drop(updated);
        let text = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![1, 3]).unwrap();
        let pointer = text.data().as_ptr();
        let Value::StringArray(updated) =
            block_on(crate::indexing::value::assign_with_plan_in_context(
                Some(&zero),
                Value::StringArray(text),
                &update,
                Value::String(Default::default()),
            ))
            .unwrap()
        else {
            panic!("string unique class")
        };
        assert_eq!(updated.data().as_ptr(), pointer);
        assert_eq!(updated.data()[1], "");
        drop(updated);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        for base in [
            Value::LogicalArray(LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap()),
            Value::CharArray(CharArray::new_row("abc")),
            Value::StringArray(
                StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![1, 3]).unwrap(),
            ),
        ] {
            let rhs = match &base {
                Value::LogicalArray(_) => Value::Bool(false),
                Value::CharArray(_) => Value::CharArray(CharArray::new_row("x")),
                _ => Value::String(Default::default()),
            };
            assert!(
                block_on(crate::indexing::value::assign_with_plan_in_context(
                    Some(&zero),
                    base.clone(),
                    &update,
                    rhs
                ))
                .is_err(),
                "shared backing must fund its replacement"
            );
        }
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        let (runtime, ledger) = managed(1_000_000);
        let both = IndexPlan::new(vec![0, 1], vec![1, 2], vec![2], 1, vec![1, 2]);
        let promoted = super::assign_logical_with_plan_host_in_context(
            Some(&runtime),
            LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap(),
            &both,
            &Value::Complex(2., -3.),
        )
        .unwrap();
        assert!(
            matches!(&promoted,Value::ComplexTensor(value) if value.materialize_f64()==vec![(2.,-3.);2])
        );
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(promoted);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[test]
    fn native_complex_single_plan_assignment_and_deletion_preserve_class() {
        let tensor =
            ComplexTensor::from_f32(vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0)], vec![1, 3])
                .unwrap();
        let plan = IndexPlan::new(vec![0, 2], vec![1, 2], vec![2], 1, vec![1, 3]);
        let Value::ComplexTensor(updated) = block_on(assign_complex_with_plan(
            tensor,
            &plan,
            &Value::Complex(1.234_567_890_123, -9.876_543_210_987),
        ))
        .expect("complex single slice assignment") else {
            panic!("expected complex tensor");
        };
        assert_eq!(updated.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            updated.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![
                (1.234_567_890_123_f64 as f32, -9.876_543_210_987_f64 as f32,),
                (2.0, -2.0),
                (1.234_567_890_123_f64 as f32, -9.876_543_210_987_f64 as f32,),
            ])
        );

        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"));
        let delete_plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 3]);
        let Value::ComplexTensor(deleted) = delete_complex_with_plan(updated, &delete_plan, &empty)
            .expect("complex single slice deletion")
        else {
            panic!("expected complex tensor");
        };
        assert_eq!(deleted.numeric_dtype(), NumericDType::F32);
        assert_eq!(deleted.shape(), [1, 2]);
        assert_eq!(
            deleted.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![
                (1.234_567_890_123_f64 as f32, -9.876_543_210_987_f64 as f32,),
                (1.234_567_890_123_f64 as f32, -9.876_543_210_987_f64 as f32,),
            ])
        );
    }

    #[test]
    fn real_tensor_rhs_fills_complex_slices_without_changing_target_class() {
        let plan = IndexPlan::new(vec![1, 2], vec![2, 1], vec![2, 1], 2, vec![4, 2]);
        let real_rhs = Value::Tensor(Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap());
        let double = ComplexTensor::new(vec![(0.0, 1.0); 8], vec![4, 2]).unwrap();
        let Value::ComplexTensor(double) =
            block_on(assign_complex_with_plan(double, &plan, &real_rhs)).unwrap()
        else {
            panic!("complex double result");
        };
        assert_eq!(double.materialize_f64()[1..3], [(5.0, 0.0), (6.0, 0.0)]);

        let single = ComplexTensor::from_f32(vec![(0.0, 1.0); 8], vec![4, 2]).unwrap();
        let Value::ComplexTensor(single) =
            block_on(assign_complex_with_plan(single, &plan, &real_rhs)).unwrap()
        else {
            panic!("complex single result");
        };
        assert_eq!(single.numeric_dtype(), NumericDType::F32);
        assert_eq!(single.materialize_f64()[1..3], [(5.0, 0.0), (6.0, 0.0)]);

        let invalid_rhs = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap());
        let error = block_on(assign_complex_with_plan(single, &plan, &invalid_rhs))
            .expect_err("real RHS shape must still match");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn typed_slice_deletion_uses_exact_storage_when_f64_mirrors_are_unavailable() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX, 3]), vec![1, 3])
            .expect("tensor");
        let plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 3]);
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty rhs"));

        let Value::Tensor(output) =
            delete_tensor_with_plan(tensor, &plan, &empty).expect("typed uint64 deletion")
        else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape(), [1, 2]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, 3]))
        );

        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I64(vec![i64::MIN, -2, i64::MAX]),
                IntegerStorage::I64(vec![1, 2, 3]),
            )
            .expect("integer complex storage"),
            vec![1, 3],
        )
        .expect("complex tensor");

        let Value::ComplexTensor(output) = delete_complex_with_plan(complex, &plan, &empty)
            .expect("typed signed complex deletion")
        else {
            panic!("expected complex tensor");
        };
        assert_eq!(output.shape(), [1, 2]);
        assert_eq!(
            output.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                    IntegerStorage::I64(vec![1, 3]),
                )
                .expect("integer complex storage")
            )
        );
    }

    #[test]
    fn integer_plan_assignment_preserves_all_typed_rhs_classes_with_poisoned_mirrors() {
        macro_rules! assert_assignment {
            ($storage:ident, $value:expr) => {{
                let tensor = Tensor::new_integer(
                    IntegerStorage::$storage(vec![Default::default(), Default::default()]),
                    vec![1, 2],
                )
                .expect("destination tensor");
                let rhs = Tensor::new_integer(IntegerStorage::$storage(vec![$value]), vec![1, 1])
                    .expect("rhs tensor");
                let plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 2]);

                let Value::Tensor(output) =
                    block_on(assign_tensor_with_plan(tensor, &plan, &Value::Tensor(rhs)))
                        .expect("assignment")
                else {
                    panic!("expected tensor");
                };
                assert_eq!(
                    output.integer_storage(),
                    Some(&IntegerStorage::$storage(vec![Default::default(), $value]))
                );
            }};
        }

        assert_assignment!(I8, i8::MIN);
        assert_assignment!(I16, i16::MIN);
        assert_assignment!(I32, i32::MIN);
        assert_assignment!(I64, i64::MIN);
        assert_assignment!(U8, u8::MAX);
        assert_assignment!(U16, u16::MAX);
        assert_assignment!(U32, u32::MAX);
        assert_assignment!(U64, u64::MAX);
    }

    #[test]
    fn real_plan_assignment_reads_typed_integer_rhs_without_mirror() {
        let tensor = Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3]).expect("tensor");
        let rhs_tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![4, 9]), vec![1, 2]).expect("rhs");
        let rhs = Value::Tensor(rhs_tensor);
        let plan = IndexPlan::new(vec![0, 2], vec![1, 2], vec![2], 1, vec![1, 3]);
        let result = block_on(assign_tensor_with_plan(tensor, &plan, &rhs)).expect("assign");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.materialize_f64(), vec![4.0, 0.0, 9.0]);
    }

    #[test]
    fn real_scalar_expansion_reads_all_typed_integer_classes_without_f64_mirrors() {
        macro_rules! assert_scalar_expansion {
            ($storage:expr, $expected:expr) => {{
                let tensor = Tensor::new_integer($storage, vec![1, 1]).expect("scalar rhs");
                let rhs = Value::Tensor(tensor);

                assert_eq!(
                    block_on(materialize_rhs_linear_real(&rhs, 3)).expect("linear expansion"),
                    vec![$expected; 3]
                );
                assert_eq!(
                    block_on(materialize_rhs_nd_real(&rhs, &[2, 2])).expect("nd expansion"),
                    vec![$expected; 4]
                );
            }};
        }

        assert_scalar_expansion!(IntegerStorage::I8(vec![-8]), -8.0);
        assert_scalar_expansion!(IntegerStorage::I16(vec![-16]), -16.0);
        assert_scalar_expansion!(IntegerStorage::I32(vec![-32]), -32.0);
        assert_scalar_expansion!(IntegerStorage::I64(vec![-64]), -64.0);
        assert_scalar_expansion!(IntegerStorage::U8(vec![8]), 8.0);
        assert_scalar_expansion!(IntegerStorage::U16(vec![16]), 16.0);
        assert_scalar_expansion!(IntegerStorage::U32(vec![32]), 32.0);
        assert_scalar_expansion!(IntegerStorage::U64(vec![64]), 64.0);
    }

    #[test]
    fn integer_plan_assignment_rejects_nonscalar_singleton_expansion() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I8(vec![0; 4]), vec![2, 2]).expect("tensor");
        let rhs = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I8(vec![5, 6]), vec![1, 2]).expect("rhs"),
        );
        let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let error = block_on(assign_tensor_with_plan(tensor, &plan, &rhs))
            .expect_err("nonscalar singleton expansion must reject");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn assignment_rhs_shape_allows_only_scalar_linear_count_or_exact_nd_shape() {
        assert_eq!(
            validated_assignment_rhs_shape(&[1, 1, 1], 1, &[2, 3]).expect("scalar"),
            vec![1, 1]
        );
        assert_eq!(
            validated_assignment_rhs_shape(&[1, 6], 6, &[6]).expect("linear count"),
            vec![6]
        );
        assert_eq!(
            validated_assignment_rhs_shape(&[2, 3, 1], 6, &[2, 3]).expect("exact shape"),
            vec![2, 3]
        );
        assert!(validated_assignment_rhs_shape(&[1, 3], 3, &[2, 3]).is_err());
        assert!(validated_assignment_rhs_shape(&[2, 1], 2, &[2, 3]).is_err());
    }

    #[test]
    fn empty_numeric_assignment_validates_rhs_without_mutating_exact_storage() {
        let plan = IndexPlan::new(vec![], vec![0], vec![0], 1, vec![1, 2]);
        let target = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            vec![1, 2],
        )
        .unwrap();
        for rhs in [
            Value::Int(runmat_value::IntValue::U64(7)),
            Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![]), vec![0, 1]).unwrap()),
        ] {
            let Value::Tensor(output) =
                block_on(assign_tensor_with_plan(target.clone(), &plan, &rhs)).unwrap()
            else {
                panic!("integer target");
            };
            assert_eq!(output.shape(), &[1, 2]);
            assert_eq!(output.integer_storage(), target.integer_storage());
        }
        for (rhs, expected) in [
            (
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                "RunMat:ShapeMismatch",
            ),
            (
                Value::String("invalid".into()),
                "RunMat:InvalidSliceAssignmentRhs",
            ),
        ] {
            let error = block_on(assign_tensor_with_plan(target.clone(), &plan, &rhs))
                .expect_err("empty selection still validates RHS");
            assert_eq!(error.identifier(), Some(expected));
            assert_eq!(
                target.integer_storage(),
                Some(&IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]))
            );
        }
    }

    #[test]
    fn empty_complex_assignment_preserves_class_specific_rhs_admission() {
        let plan = IndexPlan::new(vec![], vec![0], vec![0], 1, vec![1, 2]);
        let floating = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
        let Value::ComplexTensor(output) = block_on(assign_complex_with_plan(
            floating.clone(),
            &plan,
            &Value::Complex(7.0, 8.0),
        ))
        .unwrap() else {
            panic!("complex target");
        };
        assert_eq!(output, floating);
        let empty = Value::ComplexTensor(ComplexTensor::new(vec![], vec![0, 1]).unwrap());
        assert!(block_on(assign_complex_with_plan(floating.clone(), &plan, &empty)).is_ok());
        let incompatible = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0)], vec![1, 2]).unwrap(),
        );
        let error = block_on(assign_complex_with_plan(
            floating.clone(),
            &plan,
            &incompatible,
        ))
        .expect_err("nonempty nonscalar complex RHS");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
        let Value::ComplexTensor(output) = block_on(assign_complex_with_plan(
            floating.clone(),
            &plan,
            &Value::Int(runmat_value::IntValue::U64(1)),
        ))
        .expect("real integer scalar is a valid complex RHS") else {
            panic!("complex target");
        };
        assert_eq!(output, floating);
        let empty_real = Value::Tensor(Tensor::new(vec![], vec![0, 1]).unwrap());
        assert!(block_on(assign_complex_with_plan(
            floating.clone(),
            &plan,
            &empty_real
        ))
        .is_ok());
        let incompatible_real = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let error = block_on(assign_complex_with_plan(
            floating,
            &plan,
            &incompatible_real,
        ))
        .expect_err("nonempty nonscalar real RHS");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));

        let integer = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX, 3]),
                IntegerStorage::U64(vec![2, 4]),
            )
            .unwrap(),
            vec![1, 2],
        )
        .unwrap();
        let Value::ComplexTensor(output) = block_on(assign_complex_with_plan(
            integer.clone(),
            &plan,
            &Value::Int(runmat_value::IntValue::U64(7)),
        ))
        .unwrap() else {
            panic!("integer complex target");
        };
        assert_eq!(output, integer);
    }

    #[test]
    fn floating_logical_complex_and_string_rhs_reject_singleton_expansion() {
        let floating = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("floating"));
        let error = block_on(materialize_rhs_nd_real(&floating, &[2, 2]))
            .expect_err("floating singleton expansion must reject");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));

        let logical = Value::LogicalArray(
            runmat_value::LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical"),
        );
        let error = block_on(materialize_rhs_nd_real(&logical, &[2, 2]))
            .expect_err("logical singleton expansion must reject");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));

        let complex = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).expect("complex"),
        );
        let error = match build_complex_rhs_view(&complex, &[2, 2]) {
            Ok(_) => panic!("complex singleton expansion must reject"),
            Err(error) => error,
        };
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));

        let strings = Value::StringArray(
            runmat_value::StringArray::new(vec!["a".to_string(), "b".to_string()], vec![1, 2])
                .expect("strings"),
        );
        let error = match build_string_rhs_view(&strings, &[2, 2]) {
            Ok(_) => panic!("string singleton expansion must reject"),
            Err(error) => error,
        };
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn integer_gpu_rhs_indices_cover_linear_scalar_and_exact_nd_forms() {
        let linear = IndexPlan::new(vec![5, 1, 3], vec![1, 3], vec![3], 1, vec![2, 3]);
        assert_eq!(
            integer_gpu_rhs_indices_for_plan(&[1, 3], &linear).expect("linear indices"),
            vec![0, 1, 2]
        );
        assert_eq!(
            integer_gpu_rhs_indices_for_plan(&[1, 1], &linear).expect("scalar indices"),
            vec![0, 0, 0]
        );

        let nd = IndexPlan::new(
            vec![0, 1, 2, 3, 4, 5],
            vec![2, 3],
            vec![2, 3],
            2,
            vec![2, 3],
        );
        assert_eq!(
            integer_gpu_rhs_indices_for_plan(&[2, 3], &nd).expect("exact nd indices"),
            vec![0, 1, 2, 3, 4, 5]
        );
        assert!(integer_gpu_rhs_indices_for_plan(&[1, 3], &nd).is_err());
    }

    #[test]
    fn integer_gpu_rhs_indices_reject_incompatible_shape() {
        let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let error = integer_gpu_rhs_indices_for_plan(&[3, 1], &plan)
            .expect_err("incompatible GPU rhs shape must fail");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn integer_plan_assignment_converts_float_rhs_with_saturation() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I8(vec![0, 0]), vec![1, 2]).expect("tensor");
        let plan = IndexPlan::new(vec![0, 1], vec![1, 2], vec![2], 1, vec![1, 2]);
        let result =
            block_on(assign_tensor_with_plan(tensor, &plan, &Value::Num(300.5))).expect("assign");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I8(vec![i8::MAX, i8::MAX]))
        );
    }

    #[test]
    fn real_floating_to_complex_assignment_retains_class_and_original_alias() {
        for (dtype, source) in [
            (
                NumericDType::F32,
                Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
            ),
            (
                NumericDType::F64,
                Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
            ),
        ] {
            let original_alias = source.clone();
            let bounded = build_assignment_plan(
                &[SliceSelector::Scalar(1), SliceSelector::Scalar(2)],
                2,
                source.shape(),
            )
            .unwrap();
            let Value::ComplexTensor(updated) = block_on(assign_tensor_with_plan(
                source.clone(),
                &bounded,
                &Value::Complex(9.0, -8.0),
            ))
            .unwrap() else {
                panic!("bounded real-to-complex assignment must publish complex storage")
            };
            assert_eq!(updated.numeric_dtype(), dtype);
            assert_eq!(updated.shape(), &[2, 2]);
            assert_eq!(
                updated.materialize_f64(),
                vec![(1.0, 0.0), (2.0, 0.0), (9.0, -8.0), (4.0, 0.0)]
            );

            let growing = build_assignment_plan(
                &[SliceSelector::Scalar(3), SliceSelector::Scalar(2)],
                2,
                source.shape(),
            )
            .unwrap();
            let Value::ComplexTensor(updated) = block_on(assign_tensor_with_plan(
                source,
                &growing,
                &Value::Complex(9.0, -8.0),
            ))
            .unwrap() else {
                panic!("grown real-to-complex assignment must publish complex storage")
            };
            assert_eq!(updated.numeric_dtype(), dtype);
            assert_eq!(updated.shape(), &[3, 2]);
            assert_eq!(
                updated.materialize_f64(),
                vec![
                    (1.0, 0.0),
                    (2.0, 0.0),
                    (0.0, 0.0),
                    (3.0, 0.0),
                    (4.0, 0.0),
                    (9.0, -8.0),
                ]
            );
            assert_eq!(original_alias.numeric_dtype(), dtype);
            assert_eq!(original_alias.shape(), &[2, 2]);
            assert_eq!(original_alias.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn real_integer_plan_assignment_promotes_to_complex_exact_storage_for_every_class() {
        macro_rules! assert_promotion {
            ($storage:ident, $values:expr, $real:expr, $imag:expr) => {{
                let tensor = Tensor::new_integer(IntegerStorage::$storage($values), vec![2, 2])
                    .expect("tensor");
                let rhs_tensor = ComplexTensor::new_integer(
                    IntegerComplexStorage::new(
                        IntegerStorage::$storage(vec![$real]),
                        IntegerStorage::$storage(vec![$imag]),
                    )
                    .expect("rhs storage"),
                    vec![1, 1],
                )
                .expect("rhs");
                let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
                let result = block_on(assign_tensor_with_plan(
                    tensor,
                    &plan,
                    &Value::ComplexTensor(rhs_tensor),
                ))
                .expect("assign");
                let Value::ComplexTensor(output) = result else {
                    panic!("integer tensor should promote to complex tensor");
                };
                assert_eq!(
                    output
                        .integer_storage()
                        .map(|storage| (storage.real(), storage.imag())),
                    Some((
                        &IntegerStorage::$storage(vec![$real; 4]),
                        &IntegerStorage::$storage(vec![$imag; 4]),
                    ))
                );
            }};
        }

        assert_promotion!(I8, vec![i8::MIN, -1, 0, i8::MAX], i8::MIN, i8::MAX);
        assert_promotion!(I16, vec![i16::MIN, -1, 0, i16::MAX], i16::MIN, i16::MAX);
        assert_promotion!(I32, vec![i32::MIN, -1, 0, i32::MAX], i32::MIN, i32::MAX);
        assert_promotion!(I64, vec![i64::MIN, -1, 0, i64::MAX], i64::MIN, i64::MAX);
        assert_promotion!(U8, vec![0, 1, 2, u8::MAX], 1, u8::MAX);
        assert_promotion!(U16, vec![0, 1, 2, u16::MAX], 1, u16::MAX);
        assert_promotion!(U32, vec![0, 1, 2, u32::MAX], 1, u32::MAX);
        assert_promotion!(
            U64,
            vec![0, 1, 2, u64::MAX],
            9_223_372_036_854_775_809,
            u64::MAX
        );
    }

    #[test]
    fn real_integer_complex_promotion_rejects_shape_mismatch() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![0; 4]), vec![2, 2]).expect("tensor");
        let rhs = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::I16(vec![1, 2, 3]),
                    IntegerStorage::I16(vec![-1, -2, -3]),
                )
                .expect("rhs storage"),
                vec![1, 3],
            )
            .expect("rhs"),
        );
        let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);

        let error = block_on(assign_tensor_with_plan(tensor, &plan, &rhs))
            .expect_err("incompatible complex RHS shape must fail");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn real_integer_complex_promotion_preserves_exact_non_2d_scalar_expansion() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 3, 4]), vec![2, 1, 2])
            .expect("tensor");
        let rhs = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX]),
                    IntegerStorage::U64(vec![9_223_372_036_854_775_809]),
                )
                .expect("rhs storage"),
                vec![1, 1, 1],
            )
            .expect("rhs"),
        );
        let plan = IndexPlan::new(
            vec![0, 1, 2, 3],
            vec![2, 1, 2],
            vec![2, 1, 2],
            3,
            vec![2, 1, 2],
        );

        let result = block_on(assign_tensor_with_plan(tensor, &plan, &rhs)).expect("assign");
        let Value::ComplexTensor(output) = result else {
            panic!("integer tensor should promote to complex tensor");
        };
        assert_eq!(output.shape(), [2, 1, 2]);
        assert_eq!(
            output
                .integer_storage()
                .as_ref()
                .map(|storage| (storage.real(), storage.imag())),
            Some((
                &IntegerStorage::U64(vec![u64::MAX; 4]),
                &IntegerStorage::U64(vec![9_223_372_036_854_775_809; 4]),
            ))
        );
    }

    #[test]
    fn typed_complex_integer_plan_assignment_rejects_nonscalar_singleton_expansion() {
        let tensor = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![1, 2, 3, 4]),
                IntegerStorage::U64(vec![10, 20, 30, 40]),
            )
            .expect("storage"),
            vec![2, 2],
        )
        .expect("tensor");
        let rhs = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_808]),
                    IntegerStorage::U64(vec![7, 8]),
                )
                .expect("storage"),
                vec![1, 2],
            )
            .expect("rhs"),
        );
        let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let error = block_on(assign_complex_with_plan(tensor, &plan, &rhs))
            .expect_err("complex nonscalar singleton expansion must reject");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn typed_complex_integer_plan_assignment_rejects_shape_mismatch() {
        let tensor = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I8(vec![0; 4]),
                IntegerStorage::I8(vec![0; 4]),
            )
            .expect("storage"),
            vec![2, 2],
        )
        .expect("tensor");
        let rhs = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::I8(vec![1, 2, 3]),
                    IntegerStorage::I8(vec![4, 5, 6]),
                )
                .expect("storage"),
                vec![1, 3],
            )
            .expect("rhs"),
        );
        let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let err = block_on(assign_complex_with_plan(tensor, &plan, &rhs))
            .expect_err("shape mismatch should fail");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn typed_complex_integer_plan_assignment_converts_float_components_independently() {
        let tensor = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I8(vec![0; 2]),
                IntegerStorage::I8(vec![0; 2]),
            )
            .expect("storage"),
            vec![1, 2],
        )
        .expect("tensor");
        let plan = IndexPlan::new(vec![0, 1], vec![1, 2], vec![2], 1, vec![1, 2]);
        let result = block_on(assign_complex_with_plan(
            tensor,
            &plan,
            &Value::Complex(300.5, -300.5),
        ))
        .expect("assign");

        let Value::ComplexTensor(output) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(
            output
                .integer_storage()
                .as_ref()
                .map(|storage| (storage.real(), storage.imag())),
            Some((
                &IntegerStorage::I8(vec![i8::MAX, i8::MAX]),
                &IntegerStorage::I8(vec![i8::MIN, i8::MIN]),
            ))
        );
    }

    #[test]
    fn sparse_integer_plan_assignment_preserves_exact_values_and_last_write_wins() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let sparse =
            SparseTensor::new_integer(2, 2, vec![0, 0, 0], vec![], IntegerStorage::U64(vec![]))
                .expect("sparse");
        let rhs = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![1, 9_223_372_036_854_775_808, 3, u64::MAX]),
                vec![1, 4],
            )
            .expect("rhs"),
        );
        let plan = IndexPlan::new(vec![0, 3, 1, 3], vec![1, 4], vec![4], 1, vec![2, 2]);
        let result = block_on(assign_sparse_with_plan(sparse, &plan, &rhs)).expect("assign");

        let Value::SparseTensor(output) = result else {
            panic!("expected sparse output");
        };
        assert_eq!(output.col_ptrs, vec![0, 2, 3]);
        assert_eq!(output.row_indices, vec![0, 1, 1]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, 3, u64::MAX]))
        );
    }

    #[test]
    fn sparse_integer_plan_assignment_rejects_nonscalar_singleton_expansion() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let sparse =
            SparseTensor::new_integer(2, 2, vec![0, 0, 0], vec![], IntegerStorage::I8(vec![]))
                .expect("sparse");
        let rhs = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I8(vec![5, 6]), vec![1, 2]).expect("rhs"),
        );
        let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let error = block_on(assign_sparse_with_plan(sparse, &plan, &rhs))
            .expect_err("sparse nonscalar singleton expansion must reject");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn sparse_empty_assignment_admits_compatible_computed_empty_rhs_only() {
        let sparse = SparseTensor::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![5.0, 7.0])
            .expect("sparse target");
        let plan = IndexPlan::new(vec![], vec![0, 2], vec![0, 2], 2, vec![1, 2]);
        let compatible = Value::Tensor(Tensor::new(vec![], vec![0, 2]).unwrap());
        let Value::SparseTensor(output) =
            block_on(assign_sparse_with_plan(sparse.clone(), &plan, &compatible))
                .expect("computed empty RHS matches empty selection")
        else {
            panic!("sparse target");
        };
        assert_eq!(output, sparse);

        let incompatible = Value::Tensor(Tensor::new(vec![], vec![2, 0]).unwrap());
        let error = block_on(assign_sparse_with_plan(
            sparse.clone(),
            &plan,
            &incompatible,
        ))
        .expect_err("empty RHS dimensions must align with empty selection");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));

        let occupied = IndexPlan::new(vec![0], vec![1, 1], vec![1, 1], 2, vec![1, 2]);
        let error = block_on(assign_sparse_with_plan(
            sparse.clone(),
            &occupied,
            &compatible,
        ))
        .expect_err("nonempty sparse assignment cannot treat empty RHS as deletion");
        assert_eq!(
            error.identifier(),
            Some("RunMat:SparseAssignmentUnsupported")
        );
        let Value::SparseTensor(deleted) =
            delete_sparse_with_plan(sparse.clone(), &plan, &compatible)
                .expect("compiler-selected empty deletion remains separate")
        else {
            panic!("sparse deletion");
        };
        assert_eq!(deleted, sparse);
    }

    #[test]
    fn sparse_empty_selector_growing_plan_retains_current_runmat_shape_behavior() {
        let sparse = SparseTensor::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![5.0, 7.0])
            .expect("sparse target");
        let plan = build_sparse_assignment_plan(
            &[
                SliceSelector::Indices(vec![].into()),
                SliceSelector::Scalar(5),
            ],
            2,
            &[1, 2],
        )
        .expect("empty selector may still produce a growing plan");
        assert!(plan.indices.is_empty());
        assert_eq!(plan.selection_lengths, vec![0, 1]);
        assert_eq!(plan.base_shape, vec![1, 5]);

        // This records RunMat's existing sparse growth policy, not verified MATLAB parity.
        let Value::SparseTensor(output) = block_on(assign_sparse_with_plan(
            sparse.clone(),
            &plan,
            &Value::Num(9.0),
        ))
        .expect("scalar RHS is compatible with an empty selection") else {
            panic!("sparse output");
        };
        assert_eq!((output.rows, output.cols), (1, 5));
        assert_eq!(output.col_ptrs, vec![0, 1, 2, 2, 2, 2]);
        assert_eq!(output.row_indices, vec![0, 0]);
        assert_eq!(output.as_f64_slice(), Some(&[5.0, 7.0][..]));

        let incompatible = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let error = block_on(assign_sparse_with_plan(
            sparse.clone(),
            &plan,
            &incompatible,
        ))
        .expect_err("invalid RHS rejects before sparse expansion");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
        assert_eq!((sparse.rows, sparse.cols), (1, 2));
        assert_eq!(sparse.as_f64_slice(), Some(&[5.0, 7.0][..]));
    }

    #[test]
    fn sparse_real_plan_assignment_elides_zero_values() {
        let sparse = SparseTensor::new(2, 2, vec![0, 2, 3], vec![0, 1, 0], vec![1.0, 2.0, 3.0])
            .expect("sparse");
        let plan = IndexPlan::new(vec![0, 1], vec![2, 1], vec![2, 1], 2, vec![2, 2]);
        let result =
            block_on(assign_sparse_with_plan(sparse, &plan, &Value::Num(0.0))).expect("assign");

        let Value::SparseTensor(output) = result else {
            panic!("expected sparse output");
        };
        assert_eq!(output.col_ptrs, vec![0, 0, 1]);
        assert_eq!(output.row_indices, vec![0]);
        assert_eq!(output.as_f64_slice(), Some(&[3.0][..]));
    }

    #[test]
    fn sparse_single_plan_assignment_preserves_class_and_rounds_to_f32() {
        let sparse = SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 3.0])
            .expect("single sparse");
        let plan = IndexPlan::new(vec![0, 1], vec![2, 1], vec![2, 1], 2, vec![2, 2]);
        let rhs = Value::Tensor(Tensor::new(vec![0.0, 1.0 / 3.0], vec![2, 1]).expect("double rhs"));
        let result = block_on(assign_sparse_with_plan(sparse, &plan, &rhs)).expect("assign");

        let Value::SparseTensor(output) = result else {
            panic!("expected sparse output");
        };
        assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(output.col_ptrs, vec![0, 1, 2]);
        assert_eq!(output.row_indices, vec![1, 1]);
        assert_eq!(output.as_f32_slice(), Some(&[(1.0 / 3.0) as f32, 3.0][..]));
    }

    #[test]
    fn sparse_logical_plan_assignment_preserves_class_and_truth_conversion() {
        let sparse =
            SparseTensor::new_logical(2, 2, vec![0, 1, 2], vec![0, 1]).expect("logical sparse");
        let plan = IndexPlan::new(vec![0, 1], vec![2, 1], vec![2, 1], 2, vec![2, 2]);
        let rhs = Value::Tensor(Tensor::new(vec![0.0, -2.0], vec![2, 1]).expect("rhs"));
        let result = block_on(assign_sparse_with_plan(sparse, &plan, &rhs)).expect("assign");
        let Value::SparseTensor(output) = result else {
            panic!("expected sparse output");
        };
        assert!(output.is_logical());
        assert_eq!(output.col_ptrs, vec![0, 1, 2]);
        assert_eq!(output.row_indices, vec![1, 1]);
    }

    #[test]
    fn sparse_complex_plan_assignment_preserves_values_and_last_write_wins() {
        let sparse = SparseTensor::zeros_complex(2, 2);
        let rhs = Value::ComplexTensor(
            ComplexTensor::new(
                vec![(1.0, 2.0), (0.0, 0.0), (3.0, -4.0), (5.0, 6.0)],
                vec![1, 4],
            )
            .expect("rhs"),
        );
        let plan = IndexPlan::new(vec![0, 3, 1, 3], vec![1, 4], vec![4], 1, vec![2, 2]);
        let Value::SparseTensor(output) =
            block_on(assign_sparse_with_plan(sparse, &plan, &rhs)).expect("assign")
        else {
            panic!("expected sparse output");
        };
        assert!(output.is_complex());
        assert_eq!(output.col_ptrs, vec![0, 2, 3]);
        assert_eq!(output.row_indices, vec![0, 1, 1]);
        assert_eq!(
            output.materialize_complex_f64().expect("complex values"),
            vec![(1.0, 2.0), (3.0, -4.0), (5.0, 6.0)]
        );

        let real_rhs = Value::Tensor(Tensor::new(vec![8.0, 9.0], vec![2, 1]).unwrap());
        let plan = IndexPlan::new(vec![0, 1], vec![2, 1], vec![2, 1], 2, vec![2, 2]);
        let Value::SparseTensor(output) = block_on(assign_sparse_with_plan(
            SparseTensor::zeros_complex(2, 2),
            &plan,
            &real_rhs,
        ))
        .expect("real tensor updates complex sparse entries") else {
            panic!("sparse complex result");
        };
        assert_eq!(
            output.materialize_complex_f64().unwrap(),
            vec![(8.0, 0.0), (9.0, 0.0)]
        );

        let empty_plan = IndexPlan::new(vec![], vec![0, 1], vec![0], 1, vec![2, 2]);
        let empty_real = Value::Tensor(Tensor::new(vec![], vec![0, 1]).unwrap());
        let Value::SparseTensor(unchanged) = block_on(assign_sparse_with_plan(
            output.clone(),
            &empty_plan,
            &empty_real,
        ))
        .expect("empty real RHS keeps complex sparse source") else {
            panic!("sparse complex result");
        };
        assert_eq!(unchanged, output);
        let invalid = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let error = block_on(assign_sparse_with_plan(output, &empty_plan, &invalid))
            .expect_err("incompatible real RHS rejected before sparse update");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn integer_plan_deletion_preserves_exact_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::I64(vec![1, i64::MAX, 3]), vec![1, 3])
            .expect("tensor");
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"));
        let plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 3]);
        let result = delete_tensor_with_plan(tensor, &plan, &empty).expect("delete");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I64(vec![1, 3]))
        );
        assert_eq!(output.shape(), [1, 2]);
    }

    #[test]
    fn gpu_integer_plan_deletion_preserves_wide_uint64_storage() {
        runmat_accelerate_api::set_thread_provider(None);
        runmat_accelerate_api::clear_provider();
        runmat_accelerate::simple_provider::register_inprocess_provider();
        let provider = runmat_accelerate_api::provider().expect("test provider");
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let source = provider
            .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                data: runmat_accelerate_api::HostIntegerDataView::U64(&[
                    1,
                    9_223_372_036_854_775_808,
                    u64::MAX,
                ]),
                shape: &[1, 3],
            })
            .expect("upload integer source");
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"));
        let plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 3]);

        let Value::GpuTensor(updated) =
            block_on(delete_gpu_slice_with_plan(&source, &plan, &empty)).expect("delete")
        else {
            panic!("expected gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&updated),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        let host = block_on(provider.download_integer(&updated)).expect("download updated");
        assert_eq!(host.shape, vec![1, 2]);
        assert_eq!(
            host.data,
            runmat_accelerate_api::HostIntegerDataOwned::U64(vec![1, u64::MAX])
        );
    }

    #[test]
    fn complex_rhs_view_shape_mismatch_reports_identifier() {
        let rhs = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], vec![1, 3])
                .expect("complex tensor"),
        );
        let err = match build_complex_rhs_view(&rhs, &[2, 2]) {
            Ok(_) => panic!("shape mismatch should fail"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn complex_rhs_view_invalid_rhs_type_reports_identifier() {
        let rhs = Value::String("x".to_string().into());
        let err = match build_complex_rhs_view(&rhs, &[1]) {
            Ok(_) => panic!("non-numeric rhs should be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:InvalidSliceAssignmentRhs"));
    }

    #[test]
    fn complex_rhs_view_reads_all_typed_integer_classes_without_f64_mirrors() {
        macro_rules! assert_typed_rhs {
            ($storage:ident, $real:expr, $imag:expr) => {{
                let rhs = ComplexTensor::new_integer(
                    IntegerComplexStorage::new(
                        IntegerStorage::$storage(vec![$real]),
                        IntegerStorage::$storage(vec![$imag]),
                    )
                    .expect("typed integer components"),
                    vec![1, 1],
                )
                .expect("typed complex rhs");

                let rhs = Value::ComplexTensor(rhs);
                let view = build_complex_rhs_view(&rhs, &[1]).expect("typed complex rhs view");
                assert_eq!(view.tensor_value(0), ($real as f64, $imag as f64));
            }};
        }

        assert_typed_rhs!(I8, i8::MIN, i8::MAX);
        assert_typed_rhs!(I16, i16::MIN, i16::MAX);
        assert_typed_rhs!(I32, i32::MIN, i32::MAX);
        assert_typed_rhs!(I64, i64::MIN, i64::MAX);
        assert_typed_rhs!(U8, u8::MIN, u8::MAX);
        assert_typed_rhs!(U16, u16::MIN, u16::MAX);
        assert_typed_rhs!(U32, u32::MIN, u32::MAX);
        assert_typed_rhs!(U64, u64::MIN, u64::MAX);
    }

    #[test]
    fn string_rhs_view_shape_mismatch_reports_identifier() {
        let rhs = Value::StringArray(
            StringArray::new(
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                vec![1, 3],
            )
            .expect("string array"),
        );
        let err = match build_string_rhs_view(&rhs, &[2, 2]) {
            Ok(_) => panic!("shape mismatch should fail"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn string_cell_rhs_view_rejects_shape_data_length_mismatch() {
        let data = ["a", "b", "c"]
            .into_iter()
            .map(|text| Value::String(text.to_string().into()))
            .collect::<Vec<_>>();
        let mut cell = CellArray::new(data, 1, 3).expect("valid original payload");
        cell.shape = vec![2, 2].into();
        cell.rows = 2;
        cell.cols = 2;
        let rhs = Value::Cell(cell);
        let err = match build_string_rhs_view(&rhs, &[2, 2]) {
            Ok(_) => panic!("cell shape/data mismatch should fail"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn string_cell_rhs_uses_character_contents_not_display_layout() {
        let rhs = Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(runmat_value::CharArray::new_row("AvgOrders")),
                    Value::CharArray(runmat_value::CharArray::new_row("AvgRevenue")),
                ],
                1,
                2,
            )
            .expect("character cell row"),
        );
        let view = build_string_rhs_view(&rhs, &[2]).expect("build string RHS view");
        assert_eq!(view.count(), 2);
        assert_eq!(view.value_at(0), "AvgOrders");
        assert_eq!(view.value_at(1), "AvgRevenue");
    }

    #[test]
    fn string_rhs_view_invalid_rhs_type_reports_identifier() {
        let rhs = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("tensor"));
        let err = match build_string_rhs_view(&rhs, &[1]) {
            Ok(_) => panic!("non-string rhs should be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:InvalidSliceAssignmentRhs"));
    }

    #[test]
    fn slice_acceleration_error_mapping_reports_identifier() {
        let err = crate::indexing::resident::failure(
            "slice assign",
            anyhow::anyhow!("provider failed").into(),
        );
        assert_eq!(err.identifier(), Some("RunMat:AccelerationOperationFailed"));
    }
}
