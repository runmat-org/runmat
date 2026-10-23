//! Shared runtime dispatch over already-validated indexing plans.

use runmat_value::{Tensor, Value};

use super::plan::IndexPlan;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

mod character;
mod logical;
pub(in crate::indexing) mod primitive_output;
mod scalar;
pub(in crate::indexing) mod text;
use scalar::{assign_cell, integer_scalar_tensor, read_cell, read_scalar_struct, shape_error};

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

pub fn numeric_scalar_tensor_in_context(
    context: Option<&crate::context::RuntimeContext>,
    value: Value,
) -> Result<Tensor, RuntimeError> {
    let scalar = match value {
        Value::Num(value) => runmat_value::NumericScalar::F64(value),
        Value::Int(value) => value.into(),
        _ => return Err(invalid_base()),
    };
    let shape = super::plan::admission::shape(context, 2, |_| Ok(1))?;
    super::read_slice::payload::output::real(context, scalar.numeric_dtype(), shape, |output| {
        output.set(0, scalar)
    })
}

pub fn read_with_plan(value: Value, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    read_with_plan_in_context(context.as_ref(), value, plan)
}

pub fn read_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    value: Value,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    match value {
        Value::Tensor(value) => {
            super::read_slice::read_tensor_slice_from_plan_in_context(context, &value, plan)
        }
        Value::ComplexTensor(value) => {
            super::read_slice::read_complex_slice_from_plan_in_context(context, &value, plan)
        }
        Value::SparseTensor(value) => {
            super::read_slice::read_sparse_slice_from_plan_in_context(context, &value, plan)
        }
        Value::GpuTensor(value) => {
            super::read_slice::read_gpu_slice_from_plan_in_context(context, &value, plan)
        }
        Value::StringArray(value) => text::read(context, value.data(), plan),
        Value::LogicalArray(value) => logical::read(context, &value.data, plan),
        Value::CharArray(value) => character::read(context, &value, plan),
        Value::Cell(value) => read_cell(context, value, plan),
        Value::Struct(value) => read_scalar_struct(context, value, plan),
        Value::StructArray(value) => {
            super::structure::read_with_plan_in_context(context, &value, plan)
        }
        value @ (Value::Object(_) | Value::HandleObject(_) | Value::ObjectArray(_)) => {
            super::object::read_with_plan_in_context(context, &value, plan)
        }
        value @ (Value::Num(_) | Value::Int(_)) => read_with_plan_in_context(
            context,
            Value::Tensor(numeric_scalar_tensor_in_context(context, value)?),
            plan,
        ),
        Value::Bool(value) => logical::read(context, &[u8::from(value)], plan),
        Value::String(value) => text::read(context, std::slice::from_ref(&value), plan),
        _ => Err(invalid_base()),
    }
}

pub async fn assign_with_plan(
    value: Value,
    plan: &IndexPlan,
    rhs: Value,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    assign_with_plan_in_context(context.as_ref(), value, plan, rhs).await
}

pub async fn assign_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    value: Value,
    plan: &IndexPlan,
    rhs: Value,
) -> Result<Value, RuntimeError> {
    match value {
        Value::Tensor(value) => {
            super::write_slice::assign_tensor_with_plan_in_context(context, value, plan, &rhs).await
        }
        Value::ComplexTensor(value) => {
            super::write_slice::assign_complex_with_plan_in_context(context, value, plan, &rhs)
                .await
        }
        Value::SparseTensor(value) => {
            super::write_slice::assign_sparse_with_plan_in_context(context, value, plan, &rhs).await
        }
        Value::GpuTensor(value) => {
            super::write_slice::assign_gpu_slice_with_plan_in_context(context, &value, plan, &rhs)
                .await
        }
        Value::StringArray(value) => text::assign(context, value, plan, &rhs),
        Value::LogicalArray(value) => logical::assign(context, value, plan, &rhs),
        Value::CharArray(value) => character::assign(context, value, plan, &rhs),
        Value::Cell(value) => assign_cell(context, value, plan, &rhs),
        value @ (Value::Struct(_) | Value::StructArray(_)) => {
            super::structure::assign_with_plan_in_context(context, value, plan, rhs)
        }
        value @ (Value::Object(_) | Value::HandleObject(_) | Value::ObjectArray(_)) => {
            super::object::assign_with_plan_in_context(context, value, plan, rhs)
        }
        value @ (Value::Num(_) | Value::Int(_)) => {
            let value = numeric_scalar_tensor_in_context(context, value)?;
            super::write_slice::assign_tensor_with_plan_in_context(context, value, plan, &rhs).await
        }
        Value::Bool(value) => logical::assign_bool(context, value, plan, &rhs),
        Value::String(value) => text::assign_scalar(context, value, plan, &rhs),
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
        return assign_with_plan_in_context(context, value, plan, rhs).await;
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
            super::write_slice::delete_sparse_with_plan_in_context(context, value, plan, &rhs)
        }
        Value::GpuTensor(value) => {
            super::write_slice::delete_gpu_slice_with_plan_in_context(context, &value, plan, &rhs)
                .await
        }
        Value::Cell(value) => {
            crate::object::cell::delete_cell_with_plan_in_context(context, value, plan, &rhs)
        }
        value @ (Value::Struct(_) | Value::StructArray(_)) => {
            super::structure::delete_with_plan_in_context(context, value, plan, &rhs)
        }
        value @ (Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)) => {
            super::object::delete_with_plan_in_context(context, value, plan, &rhs)
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
