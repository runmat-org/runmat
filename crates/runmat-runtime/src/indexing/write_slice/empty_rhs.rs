use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_value::{SparseTensor, Value};

use super::rhs_shape::{checked_assignment_element_count, validate_empty_selection_rhs_shape};
use super::{mex, IndexPlan, RuntimeError};

fn validate_array_shape(shape: &[usize], len: usize, plan: &IndexPlan) -> Result<(), RuntimeError> {
    validate_empty_selection_rhs_shape(shape, len, &plan.selection_lengths)
}

fn invalid_real_rhs(rhs: &Value) -> RuntimeError {
    mex(
        "InvalidSliceAssignmentRhs",
        format!("slice assign: unsupported RHS type {rhs:?}"),
    )
}

fn validate_real_rhs(rhs: &Value, plan: &IndexPlan) -> Result<(), RuntimeError> {
    match rhs {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(()),
        Value::Tensor(tensor) => validate_array_shape(tensor.shape(), tensor.len(), plan),
        Value::LogicalArray(logical) => {
            validate_array_shape(&logical.shape, logical.data.len(), plan)
        }
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle)
                != GpuTensorStorage::ComplexInterleaved =>
        {
            let len = checked_assignment_element_count(handle.shape())?;
            validate_array_shape(handle.shape(), len, plan)
        }
        _ => Err(invalid_real_rhs(rhs)),
    }
}

pub(super) fn validate_real_or_host_complex_rhs(
    rhs: &Value,
    plan: &IndexPlan,
) -> Result<(), RuntimeError> {
    match rhs {
        Value::Complex(_, _) => Ok(()),
        Value::ComplexTensor(tensor) => validate_array_shape(tensor.shape(), tensor.len(), plan),
        _ => validate_real_rhs(rhs, plan),
    }
}

fn validate_complex_float_rhs(rhs: &Value, plan: &IndexPlan) -> Result<(), RuntimeError> {
    match rhs {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(()),
        Value::Tensor(tensor) => validate_array_shape(tensor.shape(), tensor.len(), plan),
        Value::ComplexTensor(tensor) => validate_array_shape(tensor.shape(), tensor.len(), plan),
        _ => Err(mex(
            "InvalidSliceAssignmentRhs",
            "rhs must be numeric or tensor",
        )),
    }
}

pub(super) fn validate_complex_rhs(
    rhs: &Value,
    plan: &IndexPlan,
    integer: bool,
) -> Result<(), RuntimeError> {
    if integer {
        match rhs {
            Value::Complex(_, _) => Ok(()),
            Value::ComplexTensor(tensor) => {
                validate_array_shape(tensor.shape(), tensor.len(), plan)
            }
            _ => validate_real_rhs(rhs, plan),
        }
    } else {
        validate_complex_float_rhs(rhs, plan)
    }
}

pub(super) fn validate_sparse_rhs(
    sparse: &SparseTensor,
    rhs: &Value,
    plan: &IndexPlan,
) -> Result<(), RuntimeError> {
    if sparse.is_complex() {
        validate_complex_float_rhs(rhs, plan)
    } else {
        validate_real_rhs(rhs, plan)
    }
}

pub(super) fn validate_resident_rhs(
    target: &GpuTensorHandle,
    rhs: &Value,
    plan: &IndexPlan,
) -> Result<(), RuntimeError> {
    if runmat_accelerate_api::handle_is_logical(target)
        && matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_))
    {
        return Err(mex(
            "AssignmentTypeMismatch",
            "Logical assignment requires real numeric or logical values",
        ));
    }
    if runmat_accelerate_api::handle_storage(target) == GpuTensorStorage::ComplexInterleaved {
        validate_complex_rhs(
            rhs,
            plan,
            runmat_accelerate_api::handle_integer_type(target).is_some(),
        )
    } else {
        validate_real_or_host_complex_rhs(rhs, plan)
    }
}
