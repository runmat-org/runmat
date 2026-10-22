//! Shared runtime dispatch over already-validated indexing plans.

use runmat_value::{LogicalArray, StringArray, Tensor, Value};

use super::plan::IndexPlan;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

mod scalar;
use scalar::{
    assign_cell, assign_char, assign_logical, grow_string_array, integer_scalar_tensor, read_cell,
    read_char, read_logical, read_scalar_struct, shape_error,
};

#[cfg(test)]
mod tests;

/// Ordinary owned-vector compatibility adapter for the canonical borrowed view.
pub fn shape(value: &Value) -> Option<Vec<usize>> {
    shape_borrowed(value).map(|shape| shape.as_slice().to_vec())
}

/// Preserve the indexed-value family's exact supported-class boundary while
/// borrowing its stored dimensions. In particular Symbolic remains unsupported
/// here and Object retains scalar stored geometry rather than declared payload.
pub fn shape_borrowed(
    value: &Value,
) -> Option<crate::builtins::common::shape::ValueDimensions<'_>> {
    match value {
        Value::Tensor(_)
        | Value::ComplexTensor(_)
        | Value::SparseTensor(_)
        | Value::GpuTensor(_)
        | Value::StringArray(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::Cell(_)
        | Value::StructArray(_)
        | Value::ObjectArray(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::String(_) => Some(crate::builtins::common::shape::stored_dimensions_borrowed(
            value,
        )),
        _ => None,
    }
}

/// Canonical exact scalar numeric carrier for indexed writes and deletion.
/// Integer classes stay in their native storage rather than an f64 mirror.
pub fn numeric_scalar_tensor(value: Value) -> Result<Tensor, RuntimeError> {
    match value {
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1]).map_err(shape_error),
        Value::Int(value) => integer_scalar_tensor(value),
        _ => Err(invalid_base()),
    }
}

pub fn read_with_plan(value: Value, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    match value {
        Value::Tensor(value) => super::read_slice::read_tensor_slice_from_plan(&value, plan),
        Value::ComplexTensor(value) => {
            super::read_slice::read_complex_slice_from_plan(&value, plan)
        }
        Value::SparseTensor(value) => super::read_slice::read_sparse_slice_from_plan(&value, plan),
        Value::GpuTensor(value) => super::read_slice::read_gpu_slice_from_plan(&value, plan),
        Value::StringArray(value) => super::read_slice::gather_string_slice(&value, plan),
        Value::LogicalArray(value) => read_logical(&value, plan),
        Value::CharArray(value) => read_char(&value, plan),
        Value::Cell(value) => read_cell(value, plan),
        Value::Struct(value) => read_scalar_struct(value, plan),
        Value::StructArray(value) => super::structure::read_with_plan(&value, plan),
        value @ (Value::Object(_) | Value::HandleObject(_) | Value::ObjectArray(_)) => {
            super::object::read_with_plan(&value, plan)
        }
        value @ (Value::Num(_) | Value::Int(_)) => {
            read_with_plan(Value::Tensor(numeric_scalar_tensor(value)?), plan)
        }
        Value::Bool(value) => read_logical(
            &LogicalArray::new(vec![u8::from(value)], vec![1, 1]).map_err(shape_error)?,
            plan,
        ),
        Value::String(value) => read_with_plan(
            Value::StringArray(
                StringArray::from_texts_unbounded(vec![value], vec![1, 1]).map_err(shape_error)?,
            ),
            plan,
        ),
        _ => Err(invalid_base()),
    }
}

pub async fn assign_with_plan(
    value: Value,
    plan: &IndexPlan,
    rhs: Value,
) -> Result<Value, RuntimeError> {
    match value {
        Value::Tensor(value) => {
            super::write_slice::assign_tensor_with_plan(value, plan, &rhs).await
        }
        Value::ComplexTensor(value) => {
            super::write_slice::assign_complex_with_plan(value, plan, &rhs).await
        }
        Value::SparseTensor(value) => {
            super::write_slice::assign_sparse_with_plan(value, plan, &rhs).await
        }
        Value::GpuTensor(value) => {
            super::write_slice::assign_gpu_slice_with_plan(&value, plan, &rhs).await
        }
        Value::StringArray(mut value) => {
            grow_string_array(&mut value, &plan.base_shape)?;
            let rhs = super::write_slice::build_string_rhs_view(&rhs, &plan.selection_lengths)?;
            super::write_slice::scatter_string_with_plan(&mut value, plan, &rhs)?;
            Ok(Value::StringArray(value))
        }
        Value::LogicalArray(value) => assign_logical(value, plan, &rhs),
        Value::CharArray(value) => assign_char(value, plan, &rhs),
        Value::Cell(value) => assign_cell(value, plan, &rhs),
        value @ (Value::Struct(_) | Value::StructArray(_)) => {
            super::structure::assign_with_plan(value, plan, rhs, false)
        }
        value @ (Value::Object(_) | Value::HandleObject(_) | Value::ObjectArray(_)) => {
            super::object::assign_with_plan(value, plan, rhs, false)
        }
        value @ (Value::Num(_) | Value::Int(_)) => {
            let value = numeric_scalar_tensor(value)?;
            super::write_slice::assign_tensor_with_plan(value, plan, &rhs).await
        }
        Value::Bool(value) => assign_logical(
            LogicalArray::new(vec![u8::from(value)], vec![1, 1]).map_err(shape_error)?,
            plan,
            &rhs,
        ),
        Value::String(value) => {
            let mut value =
                StringArray::from_texts_unbounded(vec![value], vec![1, 1]).map_err(shape_error)?;
            grow_string_array(&mut value, &plan.base_shape)?;
            let rhs = super::write_slice::build_string_rhs_view(&rhs, &plan.selection_lengths)?;
            super::write_slice::scatter_string_with_plan(&mut value, plan, &rhs)?;
            Ok(Value::StringArray(value))
        }
        _ => Err(invalid_base()),
    }
}

/// Execute one explicitly classified mutation over the canonical validated
/// plan. Deletion is syntax/contract authority, never inferred from RHS shape.
pub async fn mutate_with_plan(
    context: Option<&crate::context::RuntimeContext>,
    value: Value,
    plan: &IndexPlan,
    rhs: Value,
    mutation: runmat_types::PlaceMutationKind,
) -> Result<Value, RuntimeError> {
    if mutation != runmat_types::PlaceMutationKind::Delete {
        return assign_with_plan(value, plan, rhs).await;
    }
    match value {
        value @ (Value::Num(_) | Value::Int(_)) => {
            super::write_slice::delete_tensor_with_plan(numeric_scalar_tensor(value)?, plan, &rhs)
        }
        Value::Tensor(value) => super::write_slice::delete_tensor_with_plan(value, plan, &rhs),
        Value::ComplexTensor(value) => {
            super::write_slice::delete_complex_with_plan(value, plan, &rhs)
        }
        Value::SparseTensor(value) => {
            super::write_slice::delete_sparse_with_plan(value, plan, &rhs)
        }
        Value::GpuTensor(value) => {
            super::write_slice::delete_gpu_slice_with_plan(&value, plan, &rhs).await
        }
        Value::Cell(value) => {
            let indices = plan
                .indices
                .iter()
                .map(|index| {
                    usize::try_from(*index)
                        .ok()
                        .and_then(|index| index.checked_add(1))
                        .ok_or_else(|| {
                            semantic_error(
                                "IndexOutOfBounds",
                                "cell index exceeds the supported range",
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            crate::object::cell::assign_cell_paren_linear_indices_with_policy_in_context(
                context, value, &indices, &rhs, true,
            )
        }
        value @ (Value::Struct(_) | Value::StructArray(_)) => {
            super::structure::assign_with_plan(value, plan, rhs, true)
        }
        value @ (Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)) => {
            super::object::assign_with_plan(value, plan, rhs, true)
        }
        Value::LogicalArray(value) => {
            super::write_slice::delete_logical_with_plan(context, value, plan, &rhs).await
        }
        Value::Bool(value) => {
            super::write_slice::delete_bool_with_plan(context, value, plan, &rhs).await
        }
        _ => Err(invalid_base()),
    }
}

fn invalid_base() -> RuntimeError {
    semantic_error(
        "InvalidObjectSubscriptBase",
        "value does not support this subscript",
    )
}
