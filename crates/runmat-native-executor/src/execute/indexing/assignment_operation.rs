//! One owned indexed assignment, from selector decoding through receiver writeback.
//! Keeping the whole operation in one future prevents a resumed store from
//! repeating a completed planning or provider effect.

use runmat_runtime::indexing::plan::{
    build_indexed_store_plan, build_indexed_store_plan_host, IndexPlan, IndexedStorePlanMode,
};
use runmat_runtime::indexing::selectors::index_scalar_from_value_in_context;
use runmat_runtime::indexing::value as index_value;
use runmat_runtime::indexing::write_slice;
use runmat_value::{LogicalArray, Value};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::{semantic_error, value_shape, MaterializedSelectors};

pub(super) async fn assign_paren_owned(
    runtime: &runmat_runtime::context::RuntimeContext,
    base: Value,
    selectors: MaterializedSelectors,
    rhs: Value,
    delete: bool,
) -> NativeExecutorResult<Value> {
    if matches!(
        base,
        Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)
    ) {
        if let Some(indices) = object_scalar_indices(runtime, &selectors).await? {
            return if delete {
                runmat_runtime::indexing::object::delete_scalar_indices_in_context(
                    Some(runtime),
                    base,
                    &indices,
                    &rhs,
                )
            } else {
                runmat_runtime::indexing::object::assign_scalar_indices_in_context(
                    Some(runtime),
                    base,
                    &indices,
                    rhs,
                )
            }
            .map_err(NativeExecutorError::from);
        }
    }
    let plan = plan_for_assignment(&base, &selectors, !delete).await?;
    if host_assignment_eligible(&base, &rhs, delete) {
        return assign_host_with_plan(Some(runtime), base, &plan, rhs, delete);
    }
    let updated = match base {
        value @ (Value::Num(_) | Value::Int(_)) => {
            if delete {
                let tensor =
                    index_value::numeric_scalar_tensor(value).map_err(NativeExecutorError::from)?;
                write_slice::delete_tensor_with_plan(tensor, &plan, &rhs)
                    .map_err(NativeExecutorError::from)
            } else {
                index_value::assign_with_plan_in_context(Some(runtime), value, &plan, rhs)
                    .await
                    .map_err(NativeExecutorError::from)
            }
        }
        Value::Tensor(value) => {
            if delete {
                write_slice::delete_tensor_with_plan(value, &plan, &rhs)
                    .map_err(NativeExecutorError::from)
            } else {
                write_slice::assign_tensor_with_plan(value, &plan, &rhs)
                    .await
                    .map_err(NativeExecutorError::from)
            }
        }
        Value::ComplexTensor(value) => {
            if delete {
                write_slice::delete_complex_with_plan(value, &plan, &rhs)
                    .map_err(NativeExecutorError::from)
            } else {
                write_slice::assign_complex_with_plan(value, &plan, &rhs)
                    .await
                    .map_err(NativeExecutorError::from)
            }
        }
        Value::SparseTensor(value) => {
            if delete {
                write_slice::delete_sparse_with_plan_in_context(Some(runtime), value, &plan, &rhs)
                    .map_err(NativeExecutorError::from)
            } else {
                write_slice::assign_sparse_with_plan_in_context(Some(runtime), value, &plan, &rhs)
                    .await
                    .map_err(NativeExecutorError::from)
            }
        }
        Value::GpuTensor(value) => {
            if delete {
                write_slice::delete_gpu_slice_with_plan_in_context(
                    Some(runtime),
                    &value,
                    &plan,
                    &rhs,
                )
                .await
                .map_err(NativeExecutorError::from)
            } else {
                write_slice::assign_gpu_slice_with_plan_in_context(
                    Some(runtime),
                    &value,
                    &plan,
                    &rhs,
                )
                .await
                .map_err(NativeExecutorError::from)
            }
        }
        Value::Cell(value) => {
            if delete {
                return runmat_runtime::object::cell::delete_cell_with_plan_in_context(
                    Some(runtime),
                    value,
                    &plan,
                    &rhs,
                )
                .map_err(NativeExecutorError::from);
            }
            let indices = runmat_runtime::context::host_allocation::initialize_host_array(
                Some(runtime),
                plan.indices.len(),
                |ordinal| Ok(plan.indices[ordinal] as usize + 1),
            )?;
            runmat_runtime::object::cell::assign_cell_paren_linear_indices_with_policy_in_context(
                Some(runtime),
                value,
                &indices,
                &rhs,
                false,
            )
            .map_err(NativeExecutorError::from)
        }
        value @ (Value::Struct(_) | Value::StructArray(_)) => if delete {
            runmat_runtime::indexing::structure::delete_with_plan_in_context(
                Some(runtime),
                value,
                &plan,
                &rhs,
            )
        } else {
            runmat_runtime::indexing::structure::assign_with_plan_in_context(
                Some(runtime),
                value,
                &plan,
                rhs,
            )
        }
        .map_err(NativeExecutorError::from),
        value @ (Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)) => if delete {
            runmat_runtime::indexing::object::delete_with_plan_in_context(
                Some(runtime),
                value,
                &plan,
                &rhs,
            )
        } else {
            runmat_runtime::indexing::object::assign_with_plan_in_context(
                Some(runtime),
                value,
                &plan,
                rhs,
            )
        }
        .map_err(NativeExecutorError::from),
        Value::StringArray(value) if !delete => {
            if plan.indices.is_empty() {
                Ok(Value::StringArray(value))
            } else {
                write_slice::assign_string_with_plan_in_context(Some(runtime), value, &plan, &rhs)
                    .map_err(NativeExecutorError::from)
            }
        }
        Value::LogicalArray(value) => assign_logical(runtime, value, &plan, &rhs, delete).await,
        _ => Err(NativeExecutorError::from(semantic_error(
            "SliceNonTensor",
            "Indexed assignment is unsupported for this value",
        ))),
    }?;
    Ok(updated)
}

/// Only closed host representations can finish in this frame. Provider-backed
/// selectors/RHS and object protocols keep the resumable operation owner.
pub(super) fn host_assignment_eligible(base: &Value, rhs: &Value, delete: bool) -> bool {
    // Byte-native logical deletion cooperates asynchronously. It remains a
    // supported resumable operation, not a blocking host completion.
    if delete && matches!(base, Value::LogicalArray(_)) {
        return false;
    }
    if matches!(rhs, Value::GpuTensor(_)) {
        return false;
    }
    matches!(
        base,
        Value::Num(_)
            | Value::Int(_)
            | Value::Tensor(_)
            | Value::ComplexTensor(_)
            | Value::SparseTensor(_)
            | Value::Cell(_)
            | Value::Struct(_)
            | Value::StructArray(_)
            | Value::StringArray(_)
            | Value::LogicalArray(_)
    )
}

pub(super) fn plan_for_assignment_host(
    base: &Value,
    selectors: &MaterializedSelectors,
    allow_growth: bool,
) -> NativeExecutorResult<Option<IndexPlan>> {
    let shape = value_shape(base);
    build_indexed_store_plan_host(
        selectors.dims,
        selectors.colon_mask,
        selectors.end_mask,
        selectors.numeric.values(),
        &shape,
        assignment_mode(base, allow_growth),
    )
    .map_err(NativeExecutorError::from)
}

pub(super) fn assign_host_with_plan(
    runtime: Option<&runmat_runtime::context::RuntimeContext>,
    base: Value,
    plan: &IndexPlan,
    rhs: Value,
    delete: bool,
) -> NativeExecutorResult<Value> {
    match base {
        value @ (Value::Num(_) | Value::Int(_)) => {
            let value = index_value::numeric_scalar_tensor_in_context(runtime, value)
                .map_err(NativeExecutorError::from)?;
            if delete {
                write_slice::delete_tensor_with_plan(value, plan, &rhs)
            } else {
                write_slice::assign_tensor_with_plan_host(value, plan, &rhs)
            }
            .map_err(NativeExecutorError::from)
        }
        Value::Tensor(value) => if delete {
            write_slice::delete_tensor_with_plan(value, plan, &rhs)
        } else {
            write_slice::assign_tensor_with_plan_host(value, plan, &rhs)
        }
        .map_err(NativeExecutorError::from),
        Value::ComplexTensor(value) => if delete {
            write_slice::delete_complex_with_plan(value, plan, &rhs)
        } else {
            write_slice::assign_complex_with_plan_host(value, plan, &rhs)
        }
        .map_err(NativeExecutorError::from),
        Value::SparseTensor(value) => if delete {
            write_slice::delete_sparse_with_plan_in_context(runtime, value, plan, &rhs)
        } else {
            write_slice::assign_sparse_with_plan_host_in_context(runtime, value, plan, &rhs)
        }
        .map_err(NativeExecutorError::from),
        Value::Cell(value) => {
            if delete {
                return runmat_runtime::object::cell::delete_cell_with_plan_in_context(
                    runtime, value, plan, &rhs,
                )
                .map_err(NativeExecutorError::from);
            }
            let indices = runmat_runtime::context::host_allocation::initialize_host_array(
                runtime,
                plan.indices.len(),
                |ordinal| Ok(plan.indices[ordinal] as usize + 1),
            )?;
            runmat_runtime::object::cell::assign_cell_paren_linear_indices_with_policy_in_context(
                runtime, value, &indices, &rhs, false,
            )
            .map_err(NativeExecutorError::from)
        }
        value @ (Value::Struct(_) | Value::StructArray(_)) => if delete {
            runmat_runtime::indexing::structure::delete_with_plan_in_context(
                runtime, value, plan, &rhs,
            )
        } else {
            runmat_runtime::indexing::structure::assign_with_plan_in_context(
                runtime, value, plan, rhs,
            )
        }
        .map_err(NativeExecutorError::from),
        Value::StringArray(value) if !delete => {
            if plan.indices.is_empty() {
                Ok(Value::StringArray(value))
            } else {
                write_slice::assign_string_with_plan_in_context(runtime, value, plan, &rhs)
                    .map_err(NativeExecutorError::from)
            }
        }
        Value::LogicalArray(value) if !delete => {
            write_slice::assign_logical_with_plan_host_in_context(runtime, value, plan, &rhs)
                .map_err(NativeExecutorError::from)
        }
        _ => Err(NativeExecutorError::from(semantic_error(
            "SliceNonTensor",
            "Indexed assignment is unsupported for this value",
        ))),
    }
}

async fn object_scalar_indices(
    runtime: &runmat_runtime::context::RuntimeContext,
    selectors: &MaterializedSelectors,
) -> NativeExecutorResult<Option<runmat_value::ExactArrayStorage<usize>>> {
    if selectors.colon_mask != 0 || selectors.end_mask != 0 {
        return Ok(None);
    }
    let mut indices = runmat_runtime::context::host_allocation::initialize_host_array(
        Some(runtime),
        selectors.numeric.values().len(),
        |_| Ok(0usize),
    )?;
    for (ordinal, selector) in selectors.numeric.values().iter().enumerate() {
        let Some(index) = index_scalar_from_value_in_context(Some(runtime), selector).await? else {
            return Ok(None);
        };
        let index = index.positive_usize().ok_or_else(|| {
            NativeExecutorError::from(semantic_error(
                "IndexOutOfBounds",
                "Object array index is out of bounds",
            ))
        })?;
        indices
            .try_as_mut_slice_unique()
            .expect("fresh scalar index storage")[ordinal] = index;
    }
    Ok(Some(indices))
}

pub(super) async fn plan_for_assignment(
    base: &Value,
    selectors: &MaterializedSelectors,
    allow_growth: bool,
) -> NativeExecutorResult<IndexPlan> {
    let shape = value_shape(base);
    let mode = assignment_mode(base, allow_growth);
    build_indexed_store_plan(
        selectors.dims,
        selectors.colon_mask,
        selectors.end_mask,
        selectors.numeric.values(),
        &shape,
        mode,
    )
    .await
    .map_err(NativeExecutorError::from)
}

fn assignment_mode(base: &Value, allow_growth: bool) -> IndexedStorePlanMode {
    if allow_growth
        && matches!(
            base,
            Value::SparseTensor(_)
                | Value::Struct(_)
                | Value::StructArray(_)
                | Value::Tensor(_)
                | Value::Num(_)
                | Value::Int(_)
                | Value::LogicalArray(_)
                | Value::GpuTensor(_)
                | Value::ComplexTensor(_)
        )
    {
        IndexedStorePlanMode::Grow
    } else {
        IndexedStorePlanMode::Bounded
    }
}

async fn assign_logical(
    runtime: &runmat_runtime::context::RuntimeContext,
    value: LogicalArray,
    plan: &IndexPlan,
    rhs: &Value,
    delete: bool,
) -> NativeExecutorResult<Value> {
    if delete {
        write_slice::delete_logical_with_plan(Some(runtime), value, plan, rhs).await
    } else {
        write_slice::assign_logical_with_plan_in_context(Some(runtime), value, plan, rhs).await
    }
    .map_err(NativeExecutorError::from)
}
