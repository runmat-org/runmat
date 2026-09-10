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

pub fn shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Tensor(value) => Some(value.shape.clone()),
        Value::ComplexTensor(value) => Some(value.shape.clone()),
        Value::SparseTensor(value) => Some(value.shape()),
        Value::GpuTensor(value) => Some(value.shape.clone()),
        Value::StringArray(value) => Some(value.shape.clone()),
        Value::LogicalArray(value) => Some(value.shape.clone()),
        Value::CharArray(value) => Some(value.shape().to_vec()),
        Value::Cell(value) => Some(value.shape.clone()),
        Value::StructArray(value) => Some(value.shape().to_vec()),
        Value::ObjectArray(value) => Some(value.shape().to_vec()),
        Value::Struct(_) | Value::Object(_) | Value::HandleObject(_) => Some(vec![1, 1]),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => Some(vec![1, 1]),
        _ => None,
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
        Value::Num(value) => read_with_plan(
            Value::Tensor(Tensor::new(vec![value], vec![1, 1]).map_err(shape_error)?),
            plan,
        ),
        Value::Int(value) => read_with_plan(Value::Tensor(integer_scalar_tensor(value)?), plan),
        Value::Bool(value) => read_logical(
            &LogicalArray::new(vec![u8::from(value)], vec![1, 1]).map_err(shape_error)?,
            plan,
        ),
        Value::String(value) => read_with_plan(
            Value::StringArray(StringArray::new(vec![value], vec![1, 1]).map_err(shape_error)?),
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
        Value::Num(value) => {
            let value = Tensor::new(vec![value], vec![1, 1]).map_err(shape_error)?;
            super::write_slice::assign_tensor_with_plan(value, plan, &rhs).await
        }
        Value::Int(value) => {
            let value = integer_scalar_tensor(value)?;
            super::write_slice::assign_tensor_with_plan(value, plan, &rhs).await
        }
        Value::Bool(value) => assign_logical(
            LogicalArray::new(vec![u8::from(value)], vec![1, 1]).map_err(shape_error)?,
            plan,
            &rhs,
        ),
        Value::String(value) => {
            let mut value = StringArray::new(vec![value], vec![1, 1]).map_err(shape_error)?;
            grow_string_array(&mut value, &plan.base_shape)?;
            let rhs = super::write_slice::build_string_rhs_view(&rhs, &plan.selection_lengths)?;
            super::write_slice::scatter_string_with_plan(&mut value, plan, &rhs)?;
            Ok(Value::StringArray(value))
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
