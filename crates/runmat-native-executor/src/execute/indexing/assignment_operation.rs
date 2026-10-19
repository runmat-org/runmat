//! One owned indexed assignment, from selector decoding through receiver writeback.
//! Keeping the whole operation in one future prevents a resumed store from
//! repeating a completed planning or provider effect.

use runmat_runtime::indexing::plan::{
    build_indexed_store_plan, build_indexed_store_plan_host, IndexPlan, IndexedStorePlanMode,
};
use runmat_runtime::indexing::selectors::index_scalar_from_value;
use runmat_runtime::indexing::value as index_value;
use runmat_runtime::indexing::write_slice;
use runmat_value::{LogicalArray, Value};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::{semantic_error, shape_error, value_shape, MaterializedSelectors};

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
        if let Some(indices) = object_scalar_indices(&selectors).await? {
            return runmat_runtime::indexing::object::assign_scalar_indices(
                base, &indices, rhs, delete,
            )
            .map_err(NativeExecutorError::from);
        }
    }
    let plan = plan_for_assignment(&base, &selectors, !delete).await?;
    if host_assignment_eligible(&base, &rhs) {
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
                index_value::assign_with_plan(value, &plan, rhs)
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
                write_slice::delete_sparse_with_plan(value, &plan, &rhs)
                    .map_err(NativeExecutorError::from)
            } else {
                write_slice::assign_sparse_with_plan(value, &plan, &rhs)
                    .await
                    .map_err(NativeExecutorError::from)
            }
        }
        Value::GpuTensor(value) => {
            if delete {
                write_slice::delete_gpu_slice_with_plan(&value, &plan, &rhs)
                    .await
                    .map_err(NativeExecutorError::from)
            } else {
                write_slice::assign_gpu_slice_with_plan(&value, &plan, &rhs)
                    .await
                    .map_err(NativeExecutorError::from)
            }
        }
        Value::Cell(value) => {
            let indices = plan
                .indices
                .iter()
                .map(|index| *index as usize + 1)
                .collect::<Vec<_>>();
            runmat_runtime::object::cell::assign_cell_paren_linear_indices_with_policy_in_context(
                Some(runtime),
                value,
                &indices,
                &rhs,
                delete,
            )
            .map_err(NativeExecutorError::from)
        }
        value @ (Value::Struct(_) | Value::StructArray(_)) => {
            runmat_runtime::indexing::structure::assign_with_plan(value, &plan, rhs, delete)
                .map_err(NativeExecutorError::from)
        }
        value @ (Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)) => {
            runmat_runtime::indexing::object::assign_with_plan(value, &plan, rhs, delete)
                .map_err(NativeExecutorError::from)
        }
        Value::StringArray(mut value) if !delete => {
            if !plan.indices.is_empty() {
                let rhs = write_slice::build_string_rhs_view(&rhs, &plan.selection_lengths)
                    .map_err(NativeExecutorError::from)?;
                write_slice::scatter_string_with_plan(&mut value, &plan, &rhs)
                    .map_err(NativeExecutorError::from)?;
            }
            Ok(Value::StringArray(value))
        }
        Value::LogicalArray(value) => assign_logical(value, &plan, &rhs, delete).await,
        _ => Err(NativeExecutorError::from(semantic_error(
            "SliceNonTensor",
            "Indexed assignment is unsupported for this value",
        ))),
    }?;
    Ok(updated)
}

/// Only closed host representations can finish in this frame. Provider-backed
/// selectors/RHS and object protocols keep the resumable operation owner.
pub(super) fn host_assignment_eligible(base: &Value, rhs: &Value) -> bool {
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
        &selectors.numeric,
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
            let value =
                index_value::numeric_scalar_tensor(value).map_err(NativeExecutorError::from)?;
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
            write_slice::delete_sparse_with_plan(value, plan, &rhs)
        } else {
            write_slice::assign_sparse_with_plan_host(value, plan, &rhs)
        }
        .map_err(NativeExecutorError::from),
        Value::Cell(value) => {
            let indices = plan
                .indices
                .iter()
                .map(|index| *index as usize + 1)
                .collect::<Vec<_>>();
            runmat_runtime::object::cell::assign_cell_paren_linear_indices_with_policy_in_context(
                runtime, value, &indices, &rhs, delete,
            )
            .map_err(NativeExecutorError::from)
        }
        value @ (Value::Struct(_) | Value::StructArray(_)) => {
            runmat_runtime::indexing::structure::assign_with_plan(value, plan, rhs, delete)
                .map_err(NativeExecutorError::from)
        }
        Value::StringArray(mut value) if !delete => {
            if !plan.indices.is_empty() {
                let rhs = write_slice::build_string_rhs_view(&rhs, &plan.selection_lengths)
                    .map_err(NativeExecutorError::from)?;
                write_slice::scatter_string_with_plan(&mut value, plan, &rhs)
                    .map_err(NativeExecutorError::from)?;
            }
            Ok(Value::StringArray(value))
        }
        Value::LogicalArray(value) => assign_logical_host(value, plan, &rhs, delete),
        _ => Err(NativeExecutorError::from(semantic_error(
            "SliceNonTensor",
            "Indexed assignment is unsupported for this value",
        ))),
    }
}

async fn object_scalar_indices(
    selectors: &MaterializedSelectors,
) -> NativeExecutorResult<Option<Vec<usize>>> {
    if selectors.colon_mask != 0 || selectors.end_mask != 0 {
        return Ok(None);
    }
    let mut indices = Vec::with_capacity(selectors.positional.len());
    for selector in &selectors.positional {
        let Some(index) = index_scalar_from_value(selector).await? else {
            return Ok(None);
        };
        let index = index.positive_usize().ok_or_else(|| {
            NativeExecutorError::from(semantic_error(
                "IndexOutOfBounds",
                "Object array index is out of bounds",
            ))
        })?;
        indices.push(index);
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
        &selectors.numeric,
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
    value: LogicalArray,
    plan: &IndexPlan,
    rhs: &Value,
    delete: bool,
) -> NativeExecutorResult<Value> {
    let tensor = logical_as_numeric(value)?;
    let updated = if delete {
        write_slice::delete_tensor_with_plan(tensor, plan, rhs)
            .map_err(NativeExecutorError::from)?
    } else {
        write_slice::assign_tensor_with_plan(tensor, plan, rhs).await?
    };
    logical_from_numeric(updated)
}

fn assign_logical_host(
    value: LogicalArray,
    plan: &IndexPlan,
    rhs: &Value,
    delete: bool,
) -> NativeExecutorResult<Value> {
    let tensor = logical_as_numeric(value)?;
    let updated = if delete {
        write_slice::delete_tensor_with_plan(tensor, plan, rhs)
    } else {
        write_slice::assign_tensor_with_plan_host(tensor, plan, rhs)
    }
    .map_err(NativeExecutorError::from)?;
    logical_from_numeric(updated)
}

fn logical_as_numeric(value: LogicalArray) -> NativeExecutorResult<runmat_value::Tensor> {
    runmat_value::Tensor::new(
        value
            .data
            .iter()
            .map(|value| if *value != 0 { 1.0 } else { 0.0 })
            .collect(),
        value.shape,
    )
    .map_err(|error| NativeExecutorError::from(shape_error(error)))
}

fn logical_from_numeric(updated: Value) -> NativeExecutorResult<Value> {
    match updated {
        Value::Num(value) => Ok(Value::Bool(value != 0.0)),
        Value::Tensor(value) => {
            let data = value
                .materialize_f64()
                .into_iter()
                .map(|value| u8::from(value != 0.0))
                .collect();
            LogicalArray::new(data, value.shape().to_vec())
                .map(Value::LogicalArray)
                .map_err(|error| NativeExecutorError::from(shape_error(error)))
        }
        value => Ok(value),
    }
}
